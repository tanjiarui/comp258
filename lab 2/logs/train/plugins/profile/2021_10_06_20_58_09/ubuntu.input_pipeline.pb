	??b??H@??b??H@!??b??H@	e??Ot??e??Ot??!e??Ot??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??b??H@Ad?&???A????c?H@YI?+?P??rEagerKernelExecute 0*	?I+?l@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatY?e0F$??!??h??B@)m????1??]????@:Preprocessing2U
Iterator::Model::ParallelMapV2u?????!0?:6@)u?????10?:6@:Preprocessing2F
Iterator::Model??Ҥt??!????7D@)}??ݤ?1=6:Z?1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenaten??t???!ϒ?|?-@)?=	l????1B??/?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?fe?????!???<?@)?fe?????1???<?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?aL?{)??!????@)?aL?{)??1????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?:????!lIm`??M@)YL?Qԉ?1X??R@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapt?腣?!N7??}?0@)??{???s?16o?? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9d??Ot??I$2????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ad?&???Ad?&???!Ad?&???      ??!       "      ??!       *      ??!       2	????c?H@????c?H@!????c?H@:      ??!       B      ??!       J	I?+?P??I?+?P??!I?+?P??R      ??!       Z	I?+?P??I?+?P??!I?+?P??b      ??!       JCPU_ONLYYd??Ot??b q$2????X@