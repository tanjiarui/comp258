	??~K?H@??~K?H@!??~K?H@	@?W?????@?W?????!@?W?????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??~K?H@???R{??A?Ɵ?l?H@Y/Q?5?U??rEagerKernelExecute 0*	
ףp=??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat ?&?f??! ?)?U@)<?(A???1????T@:Preprocessing2F
Iterator::Model؜?gB???!?????#@)?Z'.?+??1??K?.k@:Preprocessing2U
Iterator::Model::ParallelMapV2@??r?Θ?!???`T2@)@??r?Θ?1???`T2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate\?	??b??!?????@)?_?5?!??1ɿVhH@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorzZ?????!6Ⱥ?@)zZ?????16Ⱥ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ّ?;???!|???'?V@)3?f?Ӄ?1g?N??{??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?j??P???!:,?Z???)?j??P???1:,?Z???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??N#-???!?W?X@)+?&?|?a?1???@_??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??W?????It?f?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???R{?????R{??!???R{??      ??!       "      ??!       *      ??!       2	?Ɵ?l?H@?Ɵ?l?H@!?Ɵ?l?H@:      ??!       B      ??!       J	/Q?5?U??/Q?5?U??!/Q?5?U??R      ??!       Z	/Q?5?U??/Q?5?U??!/Q?5?U??b      ??!       JCPU_ONLYY??W?????b qt?f?X@