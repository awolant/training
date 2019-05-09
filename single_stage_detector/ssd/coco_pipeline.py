# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import ctypes
import logging
from math import ceil, sqrt

import numpy as np
from collections import namedtuple
from itertools import accumulate

# DALI imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import time

def coco_anchors():
    anchors = []

    fig_size = 300
    feat_sizes = [38, 19, 10, 5, 3, 1]
    feat_count = len(feat_sizes)
    steps = [8., 16., 32., 64., 100., 300.]
    scales = [21., 45., 99., 153., 207., 261., 315.]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    fks = []
    for step in steps:
        fks.append(fig_size / step)

    anchor_idx = 0
    for idx in range(feat_count):
        sk1 = scales[idx] / fig_size
        sk2 = scales[idx + 1] / fig_size
        sk3 = sqrt(sk1 * sk2)

        all_sizes = [[sk1, sk1], [sk3, sk3]]

        for alpha in aspect_ratios[idx]:
            w = sk1 * sqrt(alpha)
            h = sk1 / sqrt(alpha)
            all_sizes.append([w, h])
            all_sizes.append([h, w])

        for sizes in all_sizes:
            w = sizes[0]
            h = sizes[1]

            for i in range(feat_sizes[idx]):
                for j in range(feat_sizes[idx]):
                    cx = (j + 0.5) / fks[idx]
                    cy = (i + 0.5) / fks[idx]

                    cx = max(min(cx, 1.), 0.)
                    cy = max(min(cy, 1.), 0.)
                    w = max(min(w, 1.), 0.)
                    h = max(min(h, 1.), 0.)

                    anchors.append(cx - 0.5 * w)
                    anchors.append(cy - 0.5 * h)
                    anchors.append(cx + 0.5 * w)
                    anchors.append(cy + 0.5 * h)

                    anchor_idx = anchor_idx + 1
    return anchors

# Defines the pipeline for a single GPU for _training_
class COCOPipeline(Pipeline):
    def __init__(
        self, 
        batch_size, 
        device_id, 
        file_root, 
        annotations_file, 
        num_gpus,
        output_fp16=False, 
        output_nhwc=False, 
        pad_output=False, 
        num_threads=1, 
        seed=15):

        super(COCOPipeline, self).__init__(
            batch_size=batch_size, 
            device_id=device_id,
            num_threads=num_threads, 
            seed = seed)

        try:
            shard_id = torch.distributed.get_rank()
        except RuntimeError:
            shard_id = 0

        self.input = ops.COCOReader(
            file_root = file_root,
            annotations_file = annotations_file,
            shard_id = shard_id,
            num_shards = num_gpus,
            ratio=True,
            ltrb=True,
            skip_empty = True,
            random_shuffle=False,
            shuffle_after_epoch=True)

        self.decode = ops.HostDecoder(device = "cpu", output_type = types.RGB)

        self.resize = ops.Resize(
            device = "gpu",
            resize_x = 300,
            resize_y = 300,
            min_filter = types.DALIInterpType.INTERP_TRIANGULAR)

        output_dtype = types.FLOAT16 if output_fp16 else types.FLOAT
        output_layout = types.NHWC if output_nhwc else types.NCHW

        mean_val = list(np.array([0.485, 0.456, 0.406]) * 255.)
        std_val = list(np.array([0.229, 0.224, 0.225]) * 255.)
        self.normalize = ops.CropMirrorNormalize(
            device="gpu", crop=(300, 300),
            mean=mean_val,
            std=std_val,
            mirror=0,
            output_dtype=output_dtype,
            output_layout=output_layout,
            pad_output=pad_output)

        self.box_encoder = ops.BoxEncoder(
            device="cpu",
            criteria=0.5,
            anchors=coco_anchors())

        # Random variables
        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()

        inputs, bboxes, labels = self.input(name='train_reader')
        images = self.decode(inputs)

        images = self.resize(images.gpu())
        images = self.normalize(images)

        bboxes, labels = self.box_encoder(bboxes, labels)


        # bboxes and images and labels on GPU
        return (images, bboxes.gpu(), labels.gpu())

to_torch_type = {
    np.dtype(np.float32) : torch.float32,
    np.dtype(np.float64) : torch.float64,
    np.dtype(np.float16) : torch.float16,
    np.dtype(np.uint8)   : torch.uint8,
    np.dtype(np.int8)    : torch.int8,
    np.dtype(np.int16)   : torch.int16,
    np.dtype(np.int32)   : torch.int32,
    np.dtype(np.int64)   : torch.int64
}

def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to pyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    """
    assert dali_tensor.shape() == list(arr.size()), \
            ("Shapes do not match: DALI tensor has size {0}"
            ", but PyTorch Tensor has size {1}".format(dali_tensor.shape(), list(arr.size())))
    #turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    dali_tensor.copy_to_external(c_type_pointer)
    return arr

DALIOutput = namedtuple('DaliOutput', ['name', 'requires_offsets', 'expected_ragged'])

class SingleDaliIterator(object):
    def __init__(self, pipeline, output_map, size, ngpu=1):
        self._pipe = pipeline
        self._size = size
        self.ngpu = ngpu

        # convert map to pair of map, need_offsets
        self._output_map = []
        self._needs_offsets = set()
        self._expected_ragged = set()
        for out in output_map:
            if isinstance(out, DALIOutput):
                self._output_map.append(out.name)
                if out.requires_offsets == True:
                    self._needs_offsets.add(out.name)
                if out.expected_ragged == True:
                    self._expected_ragged.add(out.name)
            else:
                self._output_map.append(out)

        # self._output_map = output_map
        self.batch_size = pipeline.batch_size
        self.dev_id = self._pipe.device_id

        # generate inverse map from output_map name -> dali idx
        self._inv_output_map = { name : i for i, name in enumerate(self._output_map)}

        # build the pipeline
        self._pipe.build()

        # use double-buffering
        self._data_batches = [None] * 4
        self._counter = 0
        self._current_data_batch = 0

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        self._first_batch = None
        self._first_batch = self.next()


    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter > self._size:
            raise StopIteration

        # gather outputs for this GPU
        pipe_output = self._pipe.run()

        # empty dict for each named output
        outputs = { k : [] for k in self._output_map }

        # Result of this is { output_name : dali output }
        for i, out in enumerate(pipe_output):
            outputs[self._output_map[i]].append(out)

        # Get shapes (and optional offsets) to a non-dense DALI tensor
        def convert_to_shapes(output, need_offsets=False):
            shapes = [output.at(i).shape() for i in range(len(output))]

            if need_offsets:
                # prefix sum on shape's leading dimension
                offsets = [0] + list(accumulate([o[0] for o in shapes]))

                return shapes, offsets
            else:
                return shapes, None

        # pyt_inputs are necessary to track 'tensorified' dali outputs
        pyt_inputs = {}

        # Track general outputs
        pyt_outputs = {}
        pyt_offsets = {}

        # convert DALI outputs to pytorch
        for name, out in outputs.items():
            dali_output_ind = self._inv_output_map[name]
            dali_out = pipe_output[dali_output_ind]

            # Note: There can be cases, especially with small batch sizes where even though we expect
            # a ragged tensor (and to generate offsets), the runtime batch turns out to be dense. This
            # understandably causes problems.
            # To work around this, explicitly check to see if we want offsets and use the "else" block
            # in those cases
            # There are also cases where we expect the output to be ragged, but don't want the offsets
            # (like bboxes & labels in SSD), so check for that case too
            if dali_out.is_dense_tensor() and not (name in self._needs_offsets or name in self._expected_ragged):
                if name in self._needs_offsets:
                    print('ERROR: dense tensor requires offsets')
                # Handle dense (e.g. same sized-images case)
                tensor = dali_out.as_tensor()
                shape = tensor.shape()
                # get the torch type of the image(s)
                tensor_type = to_torch_type[np.dtype(tensor.dtype())]

                # get in output device (in pytorch parlance)
                torch_gpu_device = torch.device('cuda', self.dev_id)

                # allocate the output buffer for this output in pytorch
                pyt_output = torch.zeros(shape, dtype=tensor_type, device=torch_gpu_device)

                pyt_inputs[name] = tensor
                pyt_outputs[name] = pyt_output
            else:
                need_offsets = name in self._needs_offsets
                # handle case where we have different number of values per output
                # e.g. labels, bboxes in SSD
                shapes, offsets = convert_to_shapes(dali_out, need_offsets=need_offsets)

                # get the torch type of the image(s)
                tensor_type = to_torch_type[np.dtype(dali_out.at(0).dtype())]

                # get in output device (in pytorch parlance)
                torch_gpu_device = torch.device('cuda', self.dev_id)


                # allocate output
                pyt_output = [torch.zeros(shape, dtype=tensor_type, device=torch_gpu_device) for shape in shapes]
                pyt_outputs[name] = pyt_output

                if need_offsets:
                    # allocate offsets
                    # NOTE: should these be int32 or long?
                    pyt_offset = torch.tensor(offsets, dtype=torch.int32, device=torch_gpu_device)
                    pyt_offsets[name] = pyt_offset


        # now copy each output to device

        # Copy data from DALI Tensors to torch tensors
        # NOTE: Do in more generic way
        for i, out in enumerate(pipe_output):
            output_name = self._output_map[i]

            # Same as above - need to use the same conditional or tensors aren't in the correct place and things fail
            if out.is_dense_tensor() and not (output_name in self._needs_offsets or output_name in self._expected_ragged):
                # Dense tensor - single copy
                feed_ndarray(pyt_inputs[output_name], pyt_outputs[output_name])
            else:
                # Non-dense - individual copies and cat
                for j in range(len(out)):
                    # avoid copying 0-length tensors
                    if pyt_outputs[output_name][j].shape[0] != 0:
                        feed_ndarray(out.at(j), pyt_outputs[output_name][j])
                pyt_outputs[output_name] = torch.cat(pyt_outputs[output_name])
                # NOTE: Check if we need to squeeze (last dim is 1)
                s = out.at(j).shape()
                if s[-1] == 1:
                    pyt_outputs[output_name] = pyt_outputs[output_name].squeeze(dim=len(s)-1)

        # At this point we have a map of { output_name : output_pyt_buffer } + { output_name : output_pyt_offsets }
        # Need to assemble everything and assign to data_batches
        all_outputs = []
        for k in self._output_map:
            all_outputs.append(pyt_outputs[k])
            if k in pyt_offsets:
                all_outputs.append(pyt_offsets[k])

        self._data_batches[self._current_data_batch] = tuple(all_outputs)

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 2
        self._counter += self.ngpu * self.batch_size

        return tuple(self._data_batches[copy_db_index])

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__();

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter > self._size:
            self._counter = self._counter % self._size
        else:
            logging.warning("DALI iterator does not support resetting while epoch is not finished. Ignoring...")

class NewDALICOCOIterator(object):
    def __init__(self, pipelines, output_map, size):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self._single_gpu_iterators = [SingleDaliIterator(p, output_map, size) for p in pipelines]

    def __next__(self):
        return [p.__next() for p in self._single_gpu_iterators]

    def next(self):
        return self.__next__()

    def reset(self):
        return [p.reset() for p in self._single_gpu_iterators]