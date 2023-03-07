# awesome-oneapi
An Awesome list of oneAPI projects

A curated list of awesome oneAPI and SYCL projects for AI ahd HPC. Inspired by awesome-machine-learning.

## What is oneAPI?

oneAPI is a industry standard spec that enables hetrogenuous computing
- letting you write once and support many accelerators. For more
information, you can read up at https://oneapi.io/

## Table of Contents

1. [AI - Machine Learning](#AI-\--Machine-Learning)
2. [AI - Natural Language Processing](#AI-\--Natural-Language-Proessing)
2. [AI - Natural Language Processing Chatbots](#chatbots)
3. [AI - Computer Vision](#AI-\--Computer-Vision)
4. [AI Data Science](#AI-\-Data-Science)
5. [Medical and Life Sciences](#Medical-and-Life-Sciences)
6. [Mathematics and Science](#Mathematics-and-Science)
7. [Security](#Security)
8. [Autonomous Systems](#Autonomous-Systems)
9. [Tools & Development](#Tools-and-Development)
10. [Energy](#Energy)
11. [Financial Services](#Financial-Services)
12. [Manufacturing](#Manufacturing)
13. [Tutorials](#Tutorials)
14. [Data Science](#Data-Science)


## Projects

### AI - Machine Learning
* [AccelerateDataMining](https://github.com/lkk688/DeepDataMiningLearning) - Accelerate data mining and deep learning applications via Intel onapi. Provide sample code and examples in SJSU CMPE255 Data Mining class (https://catalog.sjsu.edu/preview_course_nopop.php?catoid=12&ampcoid=58423) to demonstrate the importance of acceleration and use Intel oneAPI as one technological e
* [Performance and Portability Evaluation of the K-Means Algorithm on SYCL with CPU-GPU architectures](https://github.com/artecs-group/k-means) - This work uses the k-means algorithm to asses the performance portability of one of the most advanced implementations of the literature He-Vialle over different programming models (DPC++ CUDA OpenMP) and multi-vendor CPU-GPU architectures.


### AI - Natural Language Processing
* [SiteMana](https://github.com/SiteManaApp) - Using oneAPI powered personalization engine to converts a website visitors into a buyers
* [Gavin AI](https://github.com/Gavin-Development) - Gavin AI is a project created by Scot_Survivor (Joshua Shiells) &amp; ShmarvDogg which aims to have Englsih human like conversations through the use of AI and ML. Gavin works on the Transformer architecture however Performer &amp; FNet architectures are being investigated for better scaling.

### AI - Computer Vision

* [HIAS TassAI Facial Recognition Agent](https://github.com/AIIAL/HIAS-TassAI-Facial-Recognition-Agent) - Security is an important issue for hospitals and medical centers to consider. Today's Facial Recognition can provide ways of automating security in the medical industry reducing staffing costs and making medical facilities safer for both patients and staff.
* [Certiface Anti-Spoofing](https://github.com/cabelo/oneapi-antispoofing) - Certiface AntiSpoofing use oneAPI for fast decode video for perform liveness detection with inference. The system is capable of spotting fake faces and performing anti-face spoofing in face recognition systems.

### AI - NL

### AI - Data Science
* [GinkgoOneAPI](https://github.com/ginkgo-project/ginkgo) - In this project we want to explore the potential of having an Intel OneAPI backend for the Gingko software package: https://ginkgo-project.github.io/

* [Fighting Novel Coronavirus COVID-19 with Data Science &amp](https://github.com/azeemx/fighting_covid19_with_ds_ml.git) -  Machine Learning. In December 2019 A novel Coronavirus was found in a seafood wholesale market located in Wuhan China. On 11 February 2020 WHO announced a name for the new coronavirus disease: COVID-19. And was recognised as a pandemic on 11 March 2020.

* [Drift Detection for Edge IoT Applications](https://github.com/blackout-ai/Face_Aging_Concept_Drift) - This concept drift project is run on video and image datasets such that we can calculate an overall precision and standard error. The concept drift detection technique finds True positives and False negatives using real and virtual drift detection. 
* [Novel Applications of Transformer Models in Data Interpretation and Visualization](https://github.com/andreicozma1/oneAPI-viz) - This project explores the possibilities of using attention-based transformer models to aid in tasks related to data visualization and interpretation through the use of various Intel oneAPI toolkits such as the DNN library oneAPI AI Kit as well as oneAPI Render Kit.
* [Bioinformatic-Algorithms](https://github.com/artecs-group/nmf-dpcpp) - The computing time required to process large data matrices may become impractical even for a parallel application running on a multiprocessors cluster. NMF-DPC++ is an efficient and easy-to-use implementation of the NMF algorithm that takes advantage of the high computing performance through SYCL.
* [Pearson’s Correlation Coefficient](https://github.com/prilcool/Intel-devmesh-codeproject-two) -  Linear Regression  with DPC++ We  Implement  two  Statistical Mathematical Algorithms such as Pearson’s Correlation Coefficient &amp Linear Regression  with DPC++  and  show you how to implement this algorithms in real life in sales and marketing  to  forecast Future sales based on advertising expenditure.


### Medical and Life Sciences 

* [MCsquare](https://gitlab.com/openmcsquare/MCsquare) - Fast Monte Carlo dose calculation algorithm for the simulation of PBS proton therapy.
* [Using Intel Technologies](https://github.com/aiial/all-detection-system-for-magic-leap-1) -  Magic Leap 1 to detect Acute Lymphoblastic Leukemia (ALL) Lymphoblasts Combines Magic Leap's Spacial Computing technologies with Intel's oneAPI OpenVINO  Neural Compute Stick to provide real-time classification of Acute Lymphoblastic Leukemia Lymphoblasts in peripheral blood samples within a Mixed Reality environment.
* [Boosting epistasis detection on Intel CPU+GPU systems](https://github.com/hiperbio/cross-dpc-episdet) - This work focuses on exploring the architecture of Intel CPUs and Integrated Graphics and their heterogeneous computing potential to boost performance and energy-efficiency of epistasis detection. This will be achieved making use of OpenCL Data Parallel C++ and OpenMP programming models.

### Mathematics and Science
* [ACTS GPU R&amp](https://github.com/acts-project/traccc) - D Projects
* [ATLAS Charged Particle Seed Finding with DPC++](https://github.com/acts-project/acts) - The ATLAS Experiment is one of the general-purpose particle physics experiments built at the Large Hadron Collider (LHC) at CERN in Geneva. Its goal is to study the behavior of elementary particles at the highest energies ever produced in a laboratory help us better understand universe.
* [Homogeneous and Heterogeneous Implementations of a tridiagonal solver on Intel® Xeon® E-2176G with oneMKL getrs](https://github.com/olutosinbanjo/oneMKL_getrs.git) - Homogeneous and Heterogeneous implementations of a tridiagonal solver with oneMKL getrs 
* [Direction Field Visualization with Python](https://github.com/olutosinbanjo/direction_field) - This project demonstrates the visualization of a direction field with Python using the differential equation of a falling object as a case study.  The effectiveness of Heterogeneous Computing is also shown by exploring optimized libraries added functionalities in Intel® Distribution for Python.
* [Phase field solvers using SYCL for microstructure evolution](https://github.com/ICME-India/MicroSim) - Phase field technique is used to simulate microstructure evolution during materials processing such as 3D printing and additive manufacturing apart from traditional manufacturing techniques like welding casting etc. These non-linear PDE solvers are compute intensive and also memory intensive. #investigate

### Autonomous Systems
* [DataGAN: Leveraging Synthetic Data for Self-Driving Vehicles](https://github.com/srianumakonda/DataGAN) - Leveraging Generative Adversarial Networks to create self-driving data at scale is crucial. By emphasizing a focus on DCGANs I’m focusing on creating high-quality self-driving images that can be used to train and improve the performance of computer vision models.
* [Alice](https://github.com/intel/dffml/tree/alice/entities/alice/) - We are writing a tutorial for an open source project on how we build an AI to work on the open source project as if she were a remote developer. Bit of a self fulfilling prophecy but who doesn't love an infinite loop now and again.

### Tools and Development
* [ArrayFire - oneAPI Backend](https://github.com/arrayfire/arrayfire) - ArrayFire is a general-purpose tensor library that simplifies the process of software development for the parallel architectures found in CPUs GPUs and other hardware acceleration devices. This project is to develop a oneAPI backend to the library which currently supports CUDA OpenCL and x86.
* [Open-source Scientific Applications and Benchmarks](https://github.com/zjin-lcf/oneAPI-DirectProgramming) - This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA SYCL and OpenMP target offloading. Intel® DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program.
* [TAU Performance System](https://github.com/UO-OACISS/tau2) - The TAU Performance System® supports profiling and tracing of programs written using the Intel OneAPI. Intel OneAPI provides two interfaces for programming - OpenCL and DPC++/SYCL for CPUs and GPUs. TAU supports both - the OpenCL profiling interface and Intel Level Zero API to observe performance. 
* [TornadoVM](https://github.com/beehive-lab/TornadoVM) - TornadoVM is an open-source software technology that automatically accelerates Java programs on multi-core CPUs GPUs and FPGAs.
* [toyBrot](https://gitlab.com/VileLasagna/toyBrot) - toyBrot is a raymarching fractal generator that is used both as a  simple benchmarking tool and a study tool for parallelisation. The code is is implemented with over 10 different technologies including Intel TBB ISPC and SYCL (with support for oneAPI)
* [HPCToolKit](http://hpctoolkit.org/) - HPCToolkit is an open-source performance tool that is in some respects similar to VTune though it also works on Power and ARM architectures. It also works on NVIDIA and AMD GPUs. Our aim is to also use it for performance analysis of Intel GPUs with Intel’s OpenCL to our targets as a prelude to A0


## Tutorials
* [ROS-oneAPI](https://github.com/ftyghome/ROS-oneAPI) - A ROS package that brings intel's oneAPI to the ROS framework.This repository provides an example of vector summation on ROS using the Intel oneAPI framework. With oneAPI the summation operation can be run parallelly on CPUs GPUs and even Intel FPGA devices.
* [GWSynth](https://github.com/AndrewPastrello/GWSynth) - Synthesizes audio from gravitational waveforms produced by binary black hole inspiral-merger-ringdown simulations. The ODE solver stage of the simulation has been modified to use the Parareal parallel-in-time integration method implemented in DPC++ with Intel oneAPI.
* [Pro TBB Book code samples ported to oneAPI](https://github.com/Apress/pro-TBB) - The latest book on Threading Building Blocks (TBB) was recently published by Apress. The book comes with code samples available on GitHub. This project intends to port some of the examples to oneAPI to take advantage of the new features of this promising heterogeneous programming model.
* [dpcpp-tutorial](https://github.com/acanets/dpcpp-tutorial) - This is a collection of DPC++ sample programs that demonstrate the design analysis and optimization of DPC++ programs
* [Loop Unroll](https://github.com/shumona8/loopunrolldifferently) - Doing loop unroll using three variables

## Energy

* [A DPC++ Backend for the OCCA Portability Framework](https://github.com/libocca/occa) - OCCA—an open source portable and vendor neutral framework for parallel programming on heterogeneous platforms—is used by mission critical computational science and engineering applications of public and private sector organizations including the U.S. Department of Energy and Shell.
* [Electric Vehicles' Charging Patterns](https://github.com/Prajwal111299/Electric-Vehicle-Charging-Patterns) - Using EVs' charging data I explored when drivers are likely to plug in their cars and how much additional electricity demand will be created when the number of EVs increases.

## Financial Services


## Manufacturing
