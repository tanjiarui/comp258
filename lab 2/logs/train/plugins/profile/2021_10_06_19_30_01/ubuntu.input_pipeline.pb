	a7l[??I@a7l[??I@!a7l[??I@	??P?l????P?l??!??P?l??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:a7l[??I@L???!??A?[?~l~I@YuX????rEagerKernelExecute 0*	?Zdi@2F
Iterator::Model?Eж???!?4?
??H@)"??T2 ??1??/6;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat~???????!??)֕>@)?)?TP??1}C???8@:Preprocessing2U
Iterator::Model::ParallelMapV2;?/K;5??!????6@);?/K;5??1????6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???sE??!??A???(@)?{??Pk??1??ď?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??<????!?us<iq@)??<????1?us<iq@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????ع?!$?%?'I@)6=((E+??1??&???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?-?R\U??!???w?@)?-?R\U??1???w?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?oB!??!P???v)-@)?wak??r?1????i@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??P?l??I?x???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L???!??L???!??!L???!??      ??!       "      ??!       *      ??!       2	?[?~l~I@?[?~l~I@!?[?~l~I@:      ??!       B      ??!       J	uX????uX????!uX????R      ??!       Z	uX????uX????!uX????b      ??!       JCPU_ONLYY??P?l??b q?x???X@