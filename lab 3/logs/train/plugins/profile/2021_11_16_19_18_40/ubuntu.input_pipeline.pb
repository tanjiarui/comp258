	???J8?@???J8?@!???J8?@	xN?^1ǟ?xN?^1ǟ?!xN?^1ǟ?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???J8?@$ӡ??n??A?3??5?@Y?j???L@rEagerKernelExecute 0*	{?G???@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator5s??? @!???>?X@)5s??? @1???>?X@:Preprocessing2F
Iterator::ModelöE?@!      Y@)ܠ?[;Qr?1?ߍq???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisms????@!9~G??X@)??????q?1f~?̎???:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap}>ʈ @!ѹ???X@)??\5?a?1d?x<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9xN?^1ǟ?I????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	$ӡ??n??$ӡ??n??!$ӡ??n??      ??!       "      ??!       *      ??!       2	?3??5?@?3??5?@!?3??5?@:      ??!       B      ??!       J	?j???L@?j???L@!?j???L@R      ??!       Z	?j???L@?j???L@!?j???L@b      ??!       JCPU_ONLYYxN?^1ǟ?b q????X@