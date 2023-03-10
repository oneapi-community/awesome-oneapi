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


## Projects

### AI - Machine Learning
* [Performance and Portability Evaluation of the K-Means Algorithm on SYCL with CPU-GPU architectures](https://github.com/artecs-group/k-means) - This work uses the k-means algorithm to asses the performance portability of one of the most advanced implementations of the literature He-Vialle over different programming models (DPC++ CUDA OpenMP) and multi-vendor CPU-GPU architectures.


### AI - Natural Language Processing
* [Gavin AI](https://github.com/Gavin-Development/GavinTraining) - Gavin AI is a project created by Scot_Survivor (Joshua Shiells) ShmarvDogg which aims to have English human like conversations through the use of AI and ML. Gavin works on the Transformer architecture however Performer FNet architectures are being investigated for better scaling.

### AI - Computer Vision

* [Certiface Anti-Spoofing](https://github.com/cabelo/oneapi-antispoofing) - Certiface AntiSpoofing use oneAPI for fast decode video for perform liveness detection with inference. The system is capable of spotting fake faces and performing anti-face spoofing in face recognition systems.

### AI - NL

### AI - Data Science
* [GinkgoOneAPI](https://github.com/ginkgo-project/ginkgo) - In this project we want to explore the potential of having an Intel OneAPI backend for the Gingko software package: https://ginkgo-project.github.io/

* [HIAS TassAI Facial Recognition Agent](https://github.com/AIIAL/HIAS-TassAI-Facial-Recognition-Agent) - Security is an important issue for hospitals and medical centers to consider. Today's Facial Recognition can provide ways of automating security in the medical industry reducing staffing costs and making medical facilities safer for both patients and staff.

* [Drift Detection for Edge IoT Applications](https://github.com/blackout-ai/Face_Aging_Concept_Drift) - This concept drift project is run on video and image datasets such that we can calculate an overall precision and standard error. The concept drift detection technique finds True positives and False negatives using real and virtual drift detection. 
* [Bioinformatic-Algorithms](https://github.com/arte# Medical and Life Sciences 

* [GROMACS](https://gromacs.org) A free and open-source software suite for high-performance molecular dynamics and output analysis.

* [NAMD](https://www.ks.uiuc.edu/Research/namd/) is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems.
* [Boosting epistasis detection on Intel CPU+GPU systems](https://github.com/hiperbio/cross-dpc-episdet) - This work focuses on exploring the architecture of Intel CPUs and Integrated Graphics and their heterogeneous computing potential to boost performance and energy-efficiency of epistasis detection. This will be achieved making use of OpenCL Data Parallel C++ and OpenMP programming models.

### Mathematics and Science
* [DIVA](https://twitter.com/kitturg1/status/1586110549170847744) UC Davis optimizing volumentric rendering of scientific data using oneAPI. 
* [ACTS GPU Ramp](https://github.com/acts-project/traccc) - D Projects
* [ATLAS Charged Particle Seed Finding with DPC++](https://github.com/acts-project/acts) - The ATLAS Experiment is one of the general-purpose particle physics experiments built at the Large Hadron Collider (LHC) at CERN in Geneva. Its goal is to study the behavior of elementary particles at the highest energies ever produced in a laboratory help us better understand universe.
* [Homogeneous and Heterogeneous Implementations of a tridiagonal solver on Intel?? Xeon?? E-2176G with oneMKL getrs](https://github.com/olutosinbanjo/oneMKL_getrs.git) - Homogeneous and Heterogeneous implementations of a tridiagonal solver with oneMKL getrs 
* [Direction Field Visualization with Python](https://github.com/olutosinbanjo/direction_field) - This project demonstrates the visualization of a direction field with Python using the differential equation of a falling object as a case study.  The effectiveness of Heterogeneous Computing is also shown by exploring optimized libraries added functionalities in Intel?? Distribution for Python.

### Autonomous Systems
* [Alice](https://github.com/intel/dffml/tree/alice/entities/alice/) - We are writing a tutorial for an open source project on how we build an AI to work on the open source project as if she were a remote developer. Bit of a self fulfilling prophecy but who doesn't love an infinite loop now and again.

### Tools and Development
* [Substrate OSPray Studio](https://github.com/substrate-ospray-studio) Cloud-based rendering service, or a Rendering as a Service (RaaS), using Amazon AWS running OSPRay Studio. 
* [ArrayFire - oneAPI Backend](https://github.com/arrayfire/arrayfire) - ArrayFire is a general-purpose tensor library that simplifies the process of software development for the parallel architectures found in CPUs GPUs and other hardware acceleration devices. This project is to develop a oneAPI backend to the library which currently supports CUDA OpenCL and x86.
* [Open-source Scientific Applications and Benchmarks](https://github.com/zjin-lcf/oneAPI-DirectProgramming) - This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA SYCL and OpenMP target offloading. Intel?? DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program.
* [TAU Performance System](https://github.com/UO-OACISS/tau2) - The TAU Performance System?? supports profiling and tracing of programs written using the Intel OneAPI. Intel OneAPI provides two interfaces for programming - OpenCL and DPC++/SYCL for CPUs and GPUs. TAU supports both - the OpenCL profiling interface and Intel Level Zero API to observe performance. 
* [TornadoVM](https://github.com/beehive-lab/TornadoVM) - TornadoVM is an open-source software technology that automatically accelerates Java programs on multi-core CPUs GPUs and FPGAs.
* [toyBrot](https://gitlab.com/VileLasagna/toyBrot) - toyBrot is a raymarching fractal generator that is used both as a  simple benchmarking tool and a study tool for parallelisation. The code is is implemented with over 10 different technologies including Intel TBB ISPC and SYCL (with support for oneAPI)
* [HPCToolKit](http://hpctoolkit.org/) - HPCToolkit is an open-source performance tool that is in some respects similar to VTune though it also works on Power and ARM architectures. It also works on NVIDIA and AMD GPUs. Our aim is to also use it for performance analysis of Intel GPUs with Intel???s OpenCL to our targets as a prelude to A0


## Energy

* [A DPC++ Backend for the OCCA Portability Framework](https://github.com/libocca/occa) - OCCA???an open source portable and vendor neutral framework for parallel programming on heterogeneous platforms???is used by mission critical computational science and engineering applications of public and private sector organizations including the U.S. Department of Energy and Shell.

## Financial Services


## Manufacturing
