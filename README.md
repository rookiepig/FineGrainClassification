# FineGrainClassification

A Matlab reimplementation of the ICCV 2013 paper [Fine-Grained Categorization by Alignments](http://homes.esat.kuleuven.be/~bfernand/papers/FG_ICCV_2013.pdf). The key idea of this paper is to get features from the foreground segments which are aligned by divide the foreground into sub-regions. For more details, please check the original paper.

## Explanation of the code structure

* Folder [align](align/) contains all the Matlab codes to run the ICCV 2013 method. Run the following scripts orderly: 
  1. step1_trainEncoder.m;
  2. step2_encoding.m;
  3. step3_libsvm_kernel.m;
  4. step4_libsvm_aggre.m;
  5. step5_libsvm_traintest.m. 
(The reason for this complexity is that I need to parallelly encode all images)

* Folder [bow](bow/) is the baseline method using Bag-of-Words features, which is based on the [VLFeat bow examples](http://www.vlfeat.org/applications/caltech-101-code.html).

* Other folders are some personal experiments, e.g. latent SVM in the folder [latent](latent/). You can just ignore them.
