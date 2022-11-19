#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#

"""
## Dataset and related objects

### Dataset

The dataset class in `dlc2action` is `dataset.BehaviorDataset`. It defines all high-level interaction between
`Task`,
input data and annotations (like loading, filtering or adding pseudo-labels). It is a single class that works
for all data types and is not meant to be inherited from. All customisation happens in *store* classes instead.
Every dataset has an *input store* and an *annotation store* that perform the actual data operations.

### Store

*Stores* are defined by an abstract data handling parent class.
It is inherited from by `base_store.InputStore` and `base_store.AnnotationStore` and implementations of
these classes (for input and
annotation data, respectively, see `input_store` and `annotation_store`) are used by datasets. In other words,
adding a new dataset to `dlc2action` means
implementing a list of abstract functions for the input and annotation data.

That list of functions was created
with several **assumptions** about the structure of the data in mind. Specifically, we are assuming that you are
working with video-related information that is defined at the frame level. Videos can be separated into *clips*.
*Clips* here are elementary parts that are associated with behaviour labels. In the simplest scenario, clips are
tracks of pose estimation key points associated with single individuals, for example. Every frame of every clip
is associated with a feature vector (like key point coordinates or image pixel values) and at least some of the
frames have behaviour labels. Every video has a unique video id and every clip inside a video has a unique clip
id (clip ids in different videos don't have to be different). After this data is loaded and preprocessed,
a store can take an integer index and return an input data sample and a tuple of *original coordinates* that
can be used to map that sample to a specific place in the original data (meaning video id, clip id and frame
indices). The indexing is consistent across time and stores (the features at index 42 an an input frame
should correspond to labels at index 42 at an annotation store for the same dataset). That is checked at runtime
by comparing the original coordinates arrays of the two stores.

![image](https://i.ibb.co/Y8zc43H/data.png)

In addition, every store is defined by a tuple of *key objects* (e.g. the input data array, the original
coordinates array and a dictionary with lengths of the original videos). When these key objects are saved and a
new store is created from them, it behaves identically to the original. Finally, when initialising a dataset,
input stores are always created first and annotations stores second, if at all. If there is any information that
needs to be passed from an input store to an annotation store, it is packed in a dictionary, termed *annotation
objects*. `base_store.AnnotationStore` child classes have a `required_annotation_objects` attribute that contains
the keys that
need to be passed in any case, but you can add optional fields too.

Data is usually stored either as a `torch.Tensor` (in `base_store.AnnotationStore` instances), a
`dlc2action.utils.TensorDict` (in `base_store.InputStore` instances where all data fits in RAM) or a `numpy.ndarray`
of filenames (in `base_store.InputStore` instances with large amounts of data).
"""
