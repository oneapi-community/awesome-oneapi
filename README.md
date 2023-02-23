# awesome-oneapi
An Awesome list of oneAPI projects

A curated list of awesome oneAPI and SYCL projects for AI ahd HPC. Inspired by awesome-machine-learning.

## What is oneAPI?

oneAPI is a industry standard spec that enables hetrogenuous computing
- letting you write once and support many accelerators. For more
information, you can read up at https://oneapi.io/

## Table of Contents
1. [SYCL](#SYCL)
2. [AI](#AI)
    * [Machine Learning](#Machine-Learning)
3. [HPC](#HPC)
4. [Others](#Others)
5. [Visual](#Visual)
6. [Machine Learning](#Machine-Learning)
7. [Tutorials](#Tutorials)
8. [Libraries](#Libraries)
9. [Papers](#Papers)
10. [Blogs](#Blogs)
11. [Videos](#Videos)
12. [Announcements](#Announcements)
13. [Events](#Events)
14. [Communities](#Communities)
15. [Books](#Books)
16. [Contributions](#Contributions)

## Projects

### SYCL

* [SYCL Container](https://github.com/danchitnis/sycl-container) - A container for development SYCL code. Currently work with Intel CPU, CUDA GPU. (Intel GPU work in progress). Can be used independently of with VScode remote extension

* [toyBrot](https://gitlab.com/VileLasagna/toyBrot) - toyBrot is a raymarching fractal generator that is used both as a  simple benchmarking tool and a study tool for parallelisation. The code is is implemented with over 10 different technologies including Intel TBB ISPC and SYCL (with support for oneAPI)

* [Bioinformatic-Algorithms](https://github.com/artecs-group/nmf-dpcpp) - The computing time required to process large data matrices may become impractical even for a parallel application running on a multiprocessors cluster. NMF-DPC++ is an efficient and easy-to-use implementation of the NMF algorithm that takes advantage of the high computing performance through SYCL.

* [Distributed k-Nearest Neighbors using Locality-Sensitive Hashing and SYCL](https://github.com/SC-SGS/Distributed_GPU_LSH_using_SYCL) - In the age of AI algorithms must efficiently cope with vast data sets. We propose a performance-portable implementation of Locality-Sensitive Hashing (LSH) an approximate k-nearest neighbors algorithm using different SYCL implementations—ComputeCpp hipSYCL DPC++—supporting multiple GPUs.
* [Phase field solvers using SYCL for microstructure evolution](https://github.com/ICME-India/MicroSim) - Phase field technique is used to simulate microstructure evolution during materials processing such as 3D printing and additive manufacturing apart from traditional manufacturing techniques like welding casting etc. These non-linear PDE solvers are compute intensive and also memory intensive. #investigate

### AI
#### Machine Learning
* [Fighting Novel Coronavirus COVID-19 with Data Science &amp](https://github.com/azeemx/fighting_covid19_with_ds_ml.git) -  Machine Learning. In December 2019 A novel Coronavirus was found in a seafood wholesale market located in Wuhan China. On 11 February 2020 WHO announced a name for the new coronavirus disease: COVID-19. And was recognised as a pandemic on 11 March 2020.
* [Efficiency and productivity for decision-making on mobile SoCs](https://bitbucket.org/corbera/vi-mdp/src/oneAPI/) - Markov decision processes provide a formal framework for a computer to make decisions autonomously and intelligently when the effects of its actions are not deterministic. To solve this computationally complex problem we experiment with scheduling on a low-power heterogeneous CPU+GPU SoC platform.
* [Detecting Acute Lymphoblastic Leukemia Lymphoblasts with Tensorflow/oneAPI/OpenVINO &amp](https://github.com/aiial/hias-all-oneapi-classifier) -  Neural Compute Stick An open-source classifier programmed using the Intel® Distribution for Python* and trained using Intel® Optimization for TensorFlow*. The model is deployed on a Raspberry 4 using Intel® Distribution of OpenVINO™ Toolkit and inference is carried out using Neural Compute Stick 2.
* [Using Intel Technologies](https://github.com/aiial/all-detection-system-for-magic-leap-1) -  Magic Leap 1 to detect Acute Lymphoblastic Leukemia (ALL) Lymphoblasts Combines Magic Leap's Spacial Computing technologies with Intel's oneAPI OpenVINO  Neural Compute Stick to provide real-time classification of Acute Lymphoblastic Leukemia Lymphoblasts in peripheral blood samples within a Mixed Reality environment.
* [Using oneAPI Intel NUC  NVIDIA Jetson Nano to detect Acute Lymphoblastic Leukemia](https://github.com/AMLResearchProject/ALL-Jetson-Nano-Classifier) - The Acute Lymphoblastic Leukemia Jetson Nano Classifier is a Convolutional Neural Network developed using Intel® oneAPI AI Analytics Toolkit Intel® Optimization for Tensorflow on an Intel® NUC NUC7i7BNH  to accelerate training and TensorRT for high performance inference on NVIDIA® Jetson Nano.
* [Skin Cancer Detection](https://github.com/srianumakonda/Skin-Cancer-Detection) - Used Convolutional Neural Networks + Deep Learning algorithms to create a learned agent that can perform binary classification of malignant vs benign skin cancer.
* [DataGAN: Leveraging Synthetic Data for Self-Driving Vehicles](https://github.com/srianumakonda/DataGAN) - Leveraging Generative Adversarial Networks to create self-driving data at scale is crucial. By emphasizing a focus on DCGANs I’m focusing on creating high-quality self-driving images that can be used to train and improve the performance of computer vision models.
* [SiteMana](https://github.com/SiteManaApp) - Using oneAPI powered personalization engine to converts a website visitors into a buyers
* [Gavin AI](https://github.com/Gavin-Development) - Gavin AI is a project created by Scot_Survivor (Joshua Shiells) &amp; ShmarvDogg which aims to have Englsih human like conversations through the use of AI and ML. Gavin works on the Transformer architecture however Performer &amp; FNet architectures are being investigated for better scaling.
* [DREAMPlaceFPGA](https://github.com/rachelselinar/DREAMPlaceFPGA) - This work aims to accelerate the different stages involved in FPGA placement - global placement legalization and detailed placement using the Pytorch deep-learning toolkit. Placement in the FPGA design flow determines the physical locations of all the heterogeneous instances in the design.
* [Accelerating SeqAn](https://github.com/seqan/seqan3) - This project targets specifically the acceleration of the pairwise alignment algorithms and pattern matching algorithms contained in the SeqAn library. 
* [Drift Detection for Edge IoT Applications](https://github.com/blackout-ai/Face_Aging_Concept_Drift) - This concept drift project is run on video and image datasets such that we can calculate an overall precision and standard error. The concept drift detection technique finds True positives and False negatives using real and virtual drift detection. 
* [Alice](https://github.com/intel/dffml/tree/alice/entities/alice/) - We are writing a tutorial for an open source project on how we build an AI to work on the open source project as if she were a remote developer. Bit of a self fulfilling prophecy but who doesn't love an infinite loop now and again.
* [Exhibition Art - Shipping Cost Predictor](https://github.com/deepakjoshi2k/Machine-Learning-Exhibit-Art-Shipping-) - My project is the complete analysis of the data of a Shipping Company. The main aim is to predict the cost of shipping provided the required entries were given but along with the EDA it becomes much more than just the Cost Predictor.
* [Novel Applications of Transformer Models in Data Interpretation and Visualization](https://github.com/andreicozma1/oneAPI-viz) - This project explores the possibilities of using attention-based transformer models to aid in tasks related to data visualization and interpretation through the use of various Intel oneAPI toolkits such as the DNN library oneAPI AI Kit as well as oneAPI Render Kit.
* [RISK IDENTIFICATION IN PREDICTIVE ESTIMATION](https://github.com/QuanticTechnovations/OneAPI_RiskAnalyzer) - Significant business decisions is taken based on the outcome predicted by the ML model. Users get benefit from an unbiased statistically sound and rigorous re-validation of the prediction’s accuracy from an independent source. That is what the Predictive Risk Analyzer tool from Quantic aims to go.


### HPC

* [GinkgoOneAPI](https://github.com/ginkgo-project/ginkgo) - In this project we want to explore the potential of having an Intel OneAPI backend for the Gingko software package: https://ginkgo-project.github.io/
* [ACTS GPU R&amp](https://github.com/acts-project/traccc) - D Projects
* [TAU Performance System](https://github.com/UO-OACISS/tau2) - The TAU Performance System® supports profiling and tracing of programs written using the Intel OneAPI. Intel OneAPI provides two interfaces for programming - OpenCL and DPC++/SYCL for CPUs and GPUs. TAU supports both - the OpenCL profiling interface and Intel Level Zero API to observe performance. 
* [ArrayFire - oneAPI Backend](https://github.com/arrayfire/arrayfire) - ArrayFire is a general-purpose tensor library that simplifies the process of software development for the parallel architectures found in CPUs GPUs and other hardware acceleration devices. This project is to develop a oneAPI backend to the library which currently supports CUDA OpenCL and x86.
* [Open-source Scientific Applications and Benchmarks](https://github.com/zjin-lcf/oneAPI-DirectProgramming) - This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA SYCL and OpenMP target offloading. Intel® DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program.
* [HPCToolKit](http://hpctoolkit.org/) - HPCToolkit is an open-source performance tool that is in some respects similar to VTune though it also works on Power and ARM architectures. It also works on NVIDIA and AMD GPUs. Our aim is to also use it for performance analysis of Intel GPUs with Intel’s OpenCL to our targets as a prelude to A0
* [A Parallel Max-Miner Algorithm For An Efficient Association Rules Learning (ARL) And Knowledge Mining Using The Intel® oneAPI Toolkit](https://github.com/arthurratz/intel_max_miner_oneapi) - This project demonstrates the using of the Intel® oneAPI library to deliver a modern code in Data Parallel C++ implementing a Parallel Max-Miner algorithm to optimize the performance of the association rules learning (ARL) process
* [Boosting epistasis detection on Intel CPU+GPU systems](https://github.com/hiperbio/cross-dpc-episdet) - This work focuses on exploring the architecture of Intel CPUs and Integrated Graphics and their heterogeneous computing potential to boost performance and energy-efficiency of epistasis detection. This will be achieved making use of OpenCL Data Parallel C++ and OpenMP programming models.
* [Grid](www.github.com/paboyle/Grid) - Lattice QCD particle physics code.  #investigate
* [ATLAS Charged Particle Seed Finding with DPC++](https://github.com/acts-project/acts) - The ATLAS Experiment is one of the general-purpose particle physics experiments built at the Large Hadron Collider (LHC) at CERN in Geneva. Its goal is to study the behavior of elementary particles at the highest energies ever produced in a laboratory help us better understand universe.
* [A DPC++ Backend for the OCCA Portability Framework](https://github.com/libocca/occa) - OCCA—an open source portable and vendor neutral framework for parallel programming on heterogeneous platforms—is used by mission critical computational science and engineering applications of public and private sector organizations including the U.S. Department of Energy and Shell.
* [Homogeneous and Heterogeneous Implementations of a tridiagonal solver on Intel® Xeon® E-2176G with oneMKL getrs](https://github.com/olutosinbanjo/oneMKL_getrs.git) - Homogeneous and Heterogeneous implementations of a tridiagonal solver with oneMKL getrs 
* [Direction Field Visualization with Python](https://github.com/olutosinbanjo/direction_field) - This project demonstrates the visualization of a direction field with Python using the differential equation of a falling object as a case study.  The effectiveness of Heterogeneous Computing is also shown by exploring optimized libraries added functionalities in Intel® Distribution for Python.
* [Accelerating Irregular Codes on Elastic Dataflow Architectures](https://github.com/robertszafa/llvm-sycl-passes) - Current HLS tools fail to synthesise efficient architectures for irregular codes because they rely onstatic scheduling. Elastic dataflow techniques enable circuits to be scheduled dynamically and achievea higher performance in accelerating irregular codes.


### Others
* [Dehazing Images using Feature Attention and Knowledge Distillation](https://github.com/manncodes/dehazing-openvino) - This project presents an end-to-end feature fusion attention network (FFA-Net) to directly restore the haze-free image. To further accelerate we distill knowledge from a Teacher model( generally a model achieving SOTA) to the student model(with significantly less parameters than Teacher).
* [Spatter](https://github.com/hpcgarage/spatter) - Spatter is a new benchmark tool for assessing memory system architectures in the context of a specific category of indexed accesses known as gather and scatter. oneAPI and DevCloud are used to develop support for a oneAPI backend for Spatter that can be targeted to Intel FPGAs.
* [TornadoVM](https://github.com/beehive-lab/TornadoVM) - TornadoVM is an open-source software technology that automatically accelerates Java programs on multi-core CPUs GPUs and FPGAs.

## Visual
* [Certiface Anti-Spoofing](https://github.com/cabelo/oneapi-antispoofing) - Certiface AntiSpoofing use oneAPI for fast decode video for perform liveness detection with inference. The system is capable of spotting fake faces and performing anti-face spoofing in face recognition systems.
* [Using Intel Technologies](https://github.com/aiial/hias-all-oneapi-classifier) -  Oculus Rift to detect Acute Lymphoblastic Leukemia (ALL) Lymphoblasts Combines Oculus Rift's Virtual Reality technologies with Intel's oneAPI OpenVINO &amp; Neural Compute Stick to provide real-time classification of Acute Lymphoblastic Leukemia Lymphoblasts in peripheral blood samples within a Virtual Reality environment.
* [HIAS TassAI Facial Recognition Agent](https://github.com/AIIAL/HIAS-TassAI-Facial-Recognition-Agent) - Security is an important issue for hospitals and medical centers to consider. Today's Facial Recognition can provide ways of automating security in the medical industry reducing staffing costs and making medical facilities safer for both patients and staff.

## Tutorials
* [MCsquare](https://gitlab.com/openmcsquare/MCsquare) - Fast Monte Carlo dose calculation algorithm for the simulation of PBS proton therapy.
* [The Magic of Fractals](https://github.com/AbhiLegend/one-api-fractal-dpcpp) - In colloquial usage a fractal is "a rough or fragmented geometric shape that can be subdivided in parts each of which is (at least approximately) a reduced/size copy of the whole". The term was coined by Benoît Mandelbrot in 1975 and was derived from the Latin fractus meaning broken or fractured.
* [Raytracing From CUDA to DPC++](https://github.com/shaovoon/simple_raytracing_sycl2020) - A walkthrough of converting a code from parallel C++ ray-tracing code to CUDA and the work needed to make that CUDA code run on CPU using parallel for_each() and then converted the code to SYCL 2020 via Intel® DPC++. This is an entry competing in The Great Cross-Architecture Challenge.
* [GWSynth](https://github.com/AndrewPastrello/GWSynth) - Synthesizes audio from gravitational waveforms produced by binary black hole inspiral-merger-ringdown simulations. The ODE solver stage of the simulation has been modified to use the Parareal parallel-in-time integration method implemented in DPC++ with Intel oneAPI.
* [Parallel Realization of DCT Algorithm](https://github.com/derolol/oneapi_dct) - This project uses oneAPI to realize the parallelization of discrete cosine transform and completes the performance test of the algorithm on the DevCloud platform
* [Observing OpenMP GPU runtime events with the OMPT interface using oneAPI](https://github.com/adamtuft/ompt-target-events) - Using the OpenMP tool interface in the oneAPI OpenMP implementation to observe GPU runtime events on Intel GPUs.
* [ROS-oneAPI](https://github.com/ftyghome/ROS-oneAPI) - A ROS package that brings intel's oneAPI to the ROS framework.This repository provides an example of vector summation on ROS using the Intel oneAPI framework. With oneAPI the summation operation can be run parallelly on CPUs GPUs and even Intel FPGA devices.
* [Matrix Multiplication on Intel DevCloud using DPC++](https://github.com/KastnerRG/Read_the_docs/blob/master/docs/project6.rst) - This project is about hardware accelaration on Intel DevCloud for one API for using DPC++ on different platforms using multicore CPUGPUFPGA which uses a single programming methodology for different hardware accelarator components defined in the upgrading and for a better programmity of processor.
* [Simple Molecular Dynamics](https://github.com/VCCA2021HPC/simple-md) - Demonstrate use of SYCL for a basic molecular dynamics implementation
* [Accelerated Circuit Simulation using SYCL](https://github.com/FMarno/SYCL-LU-Decomposition) - Simulation of integrated circuits consists of solving matrix-based equations.  In this project we demonstrate the acceleration of LU decomposition as the core algorithm in solving circuits using SYCL and oneAPI on CPU and GPU. 
* [Monte Carlo Pi approximation 3 dimension](https://github.com/subarna20/pisimulation) - Monte Carlo Simulation is a broad category of computation that utilizes statistical analysis to reach a result. This sample uses the Monte Carlo Procedure to estimate the value of pi
* [Electric Vehiles' Charging Patterns](https://github.com/Prajwal111299/Electric-Vehicle-Charging-Patterns) - Using EVs' charging data I explored when drivers are likely to plug in their cars and how much additional electricity demand will be created when the number of EVs increases.
* [Performance and Portability Evaluation of the K-Means Algorithm on SYCL with CPU-GPU architectures](https://github.com/artecs-group/k-means) - This work uses the k-means algorithm to asses the performance portability of one of the most advanced implementations of the literature He-Vialle over different programming models (DPC++ CUDA OpenMP) and multi-vendor CPU-GPU architectures.
* [AccelerateDataMining](https://github.com/lkk688/DeepDataMiningLearning) - Accelerate data mining and deep learning applications via Intel onapi. Provide sample code and examples in SJSU CMPE255 Data Mining class (https://catalog.sjsu.edu/preview_course_nopop.php?catoid=12&ampcoid=58423) to demonstrate the importance of acceleration and use Intel oneAPI as one technological e
* [Pro TBB Book code samples ported to oneAPI](https://github.com/Apress/pro-TBB) - The latest book on Threading Building Blocks (TBB) was recently published by Apress. The book comes with code samples available on GitHub. This project intends to port some of the examples to oneAPI to take advantage of the new features of this promising heterogeneous programming model.
* [oneAPI example](https://github.com/fgq1994/NLP-Tutorials/blob/master/transformer.py) - transformer example
* [Computer Graphics Introduction](https://gitlab.com/bkmgit/cgintro) - Ray tracing tutorial example
* [Integral_DPCPP](https://github.com/norhidayahm/integral_dpcpp) - This project is modified from the integral project from the course "Fundamentals of Parallelism on Intel Architecture" by Dr. Andrey Vladimirov in Coursera. The codes are converted to C++ and DPC++. 
* [dpcpp-tutorial](https://github.com/acanets/dpcpp-tutorial) - This is a collection of DPC++ sample programs that demonstrate the design analysis and optimization of DPC++ programs
* [oneAPI_samples](https://github.com/Mrhuajie-wu/gw3_oneAPI_samples) - sample1:计算随机变量的数字特征，如单个变量的期望、方差，两个变量之间的协方差。同时实现两个变量之间的线性回归。sample2:卷积算法有很强的并行性，通过将每次核的矩阵乘法运算并行实现算法的并行化。sample3:展示数据在传输中以及计算中的具体情况。
* [Tweet_SentimentAnalysis_ML](https://jupyter.oneapi.devcloud.intel.com/user/u78961/lab) - Build a binary classifier to categorize the tweets in the training data as a1. Disaster/Attack/Calamity with potential loss of Life and property2. Tweets unrelated to loss of life and property3. Create a function which can take a tweet as input and classify it 
* [SolvingLinearEquations](https://github.com/LazyTigerLi/SolvingLinearEquations) - This project gives an example of solving linear equations by iterative method which is implemented using DPC++ language for Intel CPU and accelerators. However the calculation may not converge currently.
* [Loop Unroll](https://github.com/shumona8/loopunrolldifferently) - Doing loop unroll using three variables
* [Banking](https://github.com/prilcool/Intel-devmesh-codeproject-one) -  Financial Audits through parallel computing DPC++ ( Process Millions of records in seconds) Design scalable and flexible  parallel data processing and audit systems with Intel DPC++.   We show you how you can utilize DPC++  to  efficiently process millions of records in parallel in under 60 seconds . For Auditing interest  paid to each user based on the users dynamic account  balances . 
* [Pearson’s Correlation Coefficient](https://github.com/prilcool/Intel-devmesh-codeproject-two) -  Linear Regression  with DPC++ We  Implement  two  Statistical Mathematical Algorithms such as Pearson’s Correlation Coefficient &amp Linear Regression  with DPC++  and  show you how to implement this algorithms in real life in sales and marketing  to  forecast Future sales based on advertising expenditure.
* [Intel ONE API DPC++ Vector Multiplication](https://github.com/AbhiLegend/DPC-) - The vector-multiplication is a simple program that multiplies Three large vectors of integers and verifies the results. This program is implemented using C++ and Data Parallel C++ (DPC++) languages for Intel(R) CPU and accelerators.In this example you can learn how to use the most basic code in C
* [Simple Neural Network Benchmark using oneAPI and CUDA](https://github.com/Goleys/oneAPI_NeuralNetwork_Research.git) - We will create a simple Neural Network using CUDA and DPC++. Then we will use oneAPI to test our code using an Intel base workstation and compare it with the performance of an NVIDIA GPU.


## Libraries
* [oneDNN](https://github.com/oneapi-src/oneDNN) - oneAPI Deep Neural Network Library
* [oneDAL](https://github.com/oneapi-src/oneDAL) - oneAPI Data Analytics Library
* [level zero](https://github.com/oneapi-src/level-zero) - oneAPI Level Zero Specification Headers and Loader
* [oneDAL](https://github.com/oneapi-src/oneTBB) - oneAPI Threading Building Blocks
* [oneCCL](https://github.com/oneapi-src/oneCCL) - oneAPI Collective Communications Library
* [oneMKL](https://github.com/oneapi-src/oneMKL) - oneAPI Math Kernel Library

## Papers 
* [SUPRA-on-oneAPI](https://github.com/intel/supra-on-oneapi) - This paper shows the general process of migrating CUDA-based ultrasound imaging project to Intel oneAPI. we  migrated and optimized the generated DPC++ code  it can now run on Intel CPU integrated GPU(Gen9) Xe MAX GPU and Arria 10 FPGA. the B-mode imaging performance meet real-time requirement.
* [Parallel Three-Way Quicksort Using Data Parallel C++ and Intel® oneAPI Toolkits](https://github.com/arthurratz/intel_parallel_stable_sort_oneapi) - A case-study of the classical three-way quicksort performance optimization using the revolutionary Intel® oneAPI HPC Toolkit

## Blogs
[oneAPI @ Devto](https://dev.to/oneapi) - oneAPI blogs

## Videos
[GPU Performance Portability @ CppCon 2022](https://www.youtube.com/watch?v=8Cs_uI-O51s) - GPU Performance Portability Using Standard C++ with SYCL - Hugh Delaney & Rod Burns - CppCon 2022
[IWOCL 2021 - Intro to SYCL](https://www.youtube.com/watch?v=VByHaRjEQpg) - Introduction to SYCL from a session at IWOCL 2021.

## Official Announcements

## Events
* [oneAPI Events](https://oneapi.io/events) - oneAPI events both past and future.

## Community
* [/r/sycl](https://www.reddit.com/r/sycl/) - SYCL subreddit
* [/r/oneapi](https://www.reddit.com/r/oneapi/) - oneAPI subreddit
* [Stackoverflow](https://stackoverflow.com/questions/tagged/intel-oneapi) - the latest questions on oneAPI on Stackoverflow

## Books
* [Data Parallel C++: Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL](https://www.amazon.com/Data-Parallel-Mastering-Programming-Heterogeneous-ebook/dp/B08MKYHJ6R?ref_=ast_author_dp) - Learn how to accelerate C++ programs using data parallelism.
* [Pro TBB: C++ Parallel Programming with Threading Building Blocks](https://www.amazon.com/Pro-TBB-Parallel-Programming-Threading-ebook/dp/B07V5YFCMV?ref_=ast_author_dp) - This open access book is a modern guide for all C++ programmers to learn Threading Building Blocks (TBB). Written by TBB and parallel programming experts.

## Contributions


## Archive
* [Fractal Tricorn Mandel Brot](https://github.com/ranjan1977i/Fractal.git) - Fractal implementation of Conjugate of MandelBrot
* [stereo matching](https://github.com/silverfly1992/stereo-matching-dpc) - Stereo vision methods are widely used in photogrammetry automatic driving industrial inspection virtual reality and other fields. Stereo matching as a key step in stereo vision has a great influence on the accuracy and speed of measurement results. The heterogeneous acceleration technology base
* [Rocket projection using Quadratics](https://github.com/prilcool/Intel-devmesh-codeproject-four) - Use Quadratics to simulate a  rocket launch with a known equation from previous test launch  and determine  maximum rocket altitude the time it will take to reach max height The time  it will take  the rocket splash down into the ocean . so  you can take the perfect pictures  timed to perfection.
* [Cross-architecture high-order exhaustive epistasis detection on CPU and GPU devices](https://github.com/rjfnobre/crossarch-episdet) - DPC++ application for performing high-order epistasis detection searches targeting Intel DevCloud CPU and GPU nodes.
* [oneMD](https://github.com/allisterb/oneMD) - Data-parallel molecular dynamics simulator for Intel oneAPI.
* [Kick starting TensorFlow With Intel® AI Analytics Toolkit(Beta) Powered by oneAPI](https://github.com/ialimustufa/oneAPI) - Try your hands on Some amazing SOTA Models with Intel® one API in your browser without any installation;In this project I am covering 3 Models from Intel Model Zoo and evaluating them powered by Intel® oneAPI platform!
* [NASA Frontier Development Lab:  Space Resources - Localization](http://moonbench.space/) - This project developed: (1) a virtual dataset designed to test automated localization algorithms on planetary surfaces such as the Moon or Mars where GPS is unavailable; (2) a neural network approach for absolute localization by matching orbital maps with surface perspective imagery.

* [MixPose Live Yoga Stream](https://github.com/mixpose/) - MixPose is personalized AI fitness with real-time live stream yoga yoga for beginners friendly and weekly free yoga classes available.
* [Fan Efficiency Particle Simulation](https://github.com/GuMiner/DPC-experiments) - This project simulates the efficiency of various fans to determine the best candidates for rapid prototyping (3D printing). 
