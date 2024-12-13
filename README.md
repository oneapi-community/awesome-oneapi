# awesome-oneapi
An Awesome list of oneAPI projects

A curated list of awesome oneAPI and SYCL projects for solutions across industry and community. Inspired by [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning).

## What is oneAPI?

oneAPI is an open, cross-industry, standards-based, unified,
multiarchitecture, multi-vendor programming model that delivers a common
developer experience across accelerator architectures – for faster
application performance, more productivity, and greater innovation. See,
https://oneapi.io/ for more information.

## Table of Contents

1. [AI - Computer Vision](#AI-\--Computer-Vision)
2. [AI - Data Science](#AI-\--Data-Science)
3. [AI - Machine Learning](#AI-\--Machine-Learning)
4. [AI - Natural Language Processing](#AI-\--Natural-Language-Processing)
5. [AI - Frameworks and Toolkits](#AI-\--Frameworks-and-Toolkits)
6. [Autonomous Systems](#Autonomous-Systems)
7. [Data Visualization and Rendering](#Data-Visualization-and-Rendering)
8. [Energy](#Energy)
9. [Gaming](#Gaming)
10. [Manufacturing](#Manufacturing) 
11. [Mathematics and Science](#Mathematics-and-Science)
12. [Tools & Development](#Tools-and-Development)
13. [Tutorials](#Tutorials)

### AI - Computer Vision

* [DPCPP-image-Blurring-with-SYCL](https://github.com/FriedImage/DPCPP-Image-Blurring-with-SYCL) - A program developed with DPC++ SYCL for parallelizing the Image Blurring process.
* [visionicpp](https://github.com/codeplaysoftware/visioncpp) - A machine vision library written in SYCL and C++ that shows performance-portable implementation of graph algorithms

### AI - Data Science

* [Boosting epistasis detection on Intel CPU+GPU systems](https://github.com/hiperbio/cross-dpc-episdet) - This work focuses on exploring the architecture of Intel CPUs and Integrated Graphics and their heterogeneous computing potential to boost performance and energy-efficiency of epistasis detection. This will be achieved making use of OpenCL Data Parallel C++ and OpenMP programming models.
* [root-experimental](https://github.com/jolly-chen/root/tree/gpu_histogram_bulk/tree/experimental) - Jolly Chen's fork of root.cern demnostrating porting RDataFrame to SYCL from CUDA.

### AI - Machine Learning

* [Performance and Portability Evaluation of the K-Means Algorithm on SYCL with CPU-GPU architectures](https://github.com/artecs-group/k-means) - This work uses the k-means algorithm to asses the performance portability of one of the most advanced implementations of the literature He-Vialle over different programming models (DPC++ CUDA OpenMP) and multi-vendor CPU-GPU architectures.
* [dpcpp-svm](https://github.com/floatshadow/dpcpp-svm) - A DPC++ version of ThunderSVM. The mission of ThunderSVM is to help users easily and efficiently apply SVMs to solve problems. ThunderSVM exploits GPU and multi-core CPUs to achieve high efficiency. 
* [PLSSVM](https://github.com/SC-SGS/PLSSVM) - Implementation of a parallel least squares support vector machine using multiple backends for different GPU vendors.

### AI - Natural Language Processing

* [CTranslate2](https://github.com/OpenNMT/CTranslate2) - CTranslate2 is a C and Python library that optimizes inference with transformer models, supporting models trained in various frameworks. It implements various performance optimization techniques such as weights quantization, layers fusion, batch reordering, and more for benchmarks of transformer models on CPU and GPU.
* [hachi](https://github.com/ramanlabs-in/hachi) - Hachi is a locally hosted web app that enables natural language search for videos and images, using an AI-based machine learning model powered by OpenAI CLIP.
* [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) - Whisper ctranslate2 is a command-line client based on ctranslate2, compatible with original OpenAI client.

### AI - Frameworks and Toolkits

* [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) - The Eclipse DeepLearning4J ecosystem supports all the needs for JVM-based deep learning applications with various libraries
* [deeplearning4j-examples](https://github.com/deeplearning4j/deeplearning4j-examples) - The Eclipse Deeplearning4j (DL4J) ecosystem is a set of projects that supports all the needs of a JVM-based deep learning application.
* [DeepRec](https://github.com/DeepRec-AI/DeepRec) - DeepRec is a recommendation deep learning framework based on TensorFlow, which has been developed since 2016 and supports core businesses such as Taobao search recommendation and advertising.
* [dlstreamer](https://github.com/dlstreamer/dlstreamer) - The Intel Deep Learning Streamer is an open source streaming media analytics framework based on the GStreamer multimedia framework. It is optimized for performance and functional interoperability between GStreamer plugins built on various backend libraries, with support for over 70 pre-trained models for various use cases.
* [flashlight](https://github.com/flashlight/flashlight) - Flashlight is a machine learning library written in C and created by Facebook AI Research. It features internal APIs for tensor computation, high performance defaults using just-in-time kernel compilation, and scalability
* [intel-extension-for-tensorflow](https://github.com/intel/intel-extension-for-tensorflow) - Intel Extension for TensorFlow is a plugin based on TensorFlow PluggableDevice, which aims to bring devices such as Intel XPU, GPU, and CPU into TensorFlow.
* [intel-extension-for-transformers](https://github.com/intel/intel-extension-for-transformers) - Intel Extension for Transformers is a toolkit designed to efficiently accelerate transformer-based models on Intel platforms, optimized for 4th gen Intel Xeon Scalable Processor (codename Sapphire Rapids).
* [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch) - Intel Extension for PyTorch provides features optimizations for an extra performance boost on Intel hardware including CPUs and Discrete GPUs and offers easy GPU acceleration for Intel Discrete GPUs with PyTorch.
* [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) - KernelAbstractions (KA) is a package that enables you to write GPU-like kernels targetting different execution backends.  
* [neural-compressor](https://github.com/intel/neural-compressor) - Intel Neural Compressor is an open-source Python library for applying popular model compression techniques, such as pruning, quantization, sparsity, and distillation, on all mainstream deep learning frameworks and Intel extensions.
* [optimum-intel](https://github.com/huggingface/optimum-intel) - Optimum Intel is an interface between the Transformers and Diffusers libraries and Intel's different tools and libraries that help accelerate end-to-end pipelines on Intel architectures.
* [portDNN](https://github.com/codeplaysoftware/portDNN) - portDNN is a library implementing neural network algorithms written using SYCL.
* [PPLNN](https://github.com/openppl-public/ppl.nn) - PPLNN, which is short for "PPLNN is a Primitive Library for Neural Network", is a high-performance deep-learning inference engine for efficient AI inferencing. It can run various ONNX models and has better support for OpenMMLab.
* [pynufft](https://github.com/jyhmiinlin/pynufft) - The pynufft library is a Python package for non-uniform fast Fourier transform, based on a min-max interpolator, with experimental support for CuPy, PyTorch, and TensorFlow Eager mode
* [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) - Intel r Extension for scikit learn is a free AI accelerator that can accelerate existing scikit learn code without the need to change the existing code. It offers patching and replacing the stock scikit learn algorithms with their optimized versions provided by the extension, which results in over 10-100x acceleration across a variety of applications.
* [shumai](https://github.com/facebookresearch/shumai) - The Shumai project is a differentiable tensor library for TypeScript and JavaScript built with Bun and Flashlight. It provides standard array utilities, gradients, and supported operators.
* [webnn-native](https://github.com/webmachinelearning/webnn-native)- WebNN Native is an implementation of the Web Neural Network API, providing building blocks, headers, and backends for ML platforms including DirectML, OpenVINO, and XNNPACK.
* [ZenDNN](https://github.com/amd/ZenDNN) - Zen deep neural network library ZendNN is a powerful library for deep learning inference applications on AMD CPUs. It includes APIs for basic neural network building blocks and is optimized for AMD CPUs.

### Autonomous Systems

* [FastChat](https://github.com/lm-sys/FastChat) - FastChat is an open platform for training, serving, and evaluating large language model based chatbots.

### Data Visualization and Rendering

* [Blender](https://github.com/blender) - Blender is the free and open source 3D creation suite. It supports the entirety of the 3D pipeline-modeling, rigging, animation, simulation, rendering, compositing, motion tracking and video editing.
* [Brayns](https://github.com/BlueBrain/Brayns) - Brayns is a large scientific visualization platform based on CPU ray tracing, using an extension plugin architecture. It comes with several pre-made plugins, such as CircuitExplorer and MoleculeExplorer, and requires several dependencies to build
* [ChameleonRT](https://github.com/Twinklebear/ChameleonRT) - ChameleonRT is an example path tracer that runs on multiple ray tracing backends including Embree, SYCL, DXR, Optix, Vulkan, Metal, and Ospray.
* [embree](https://github.com/embree/embree) - Embree is a high performance ray tracing library developed by Intel that targets graphics application developers to improve the performance of photo-realistic rendering applications. It includes various primitive types such as triangles, quads, grids, and curve primitives, and supports dynamic scenes. Embree also offers support for both CPUs and GPUs, while maintaining one code base to improve productivity and eliminate inconsistencies between the two versions of the renderer.
* [fresnel](https://github.com/glotzerlab/fresnel) - Fresnel is a Python library for path tracing that can be used to generate high quality images in real time.
* [f3d](https://github.com/f3d-app/f3d) - F3D is a fast and minimalist 3D viewer that supports multiple file formats and can show animations, supporting thumbnails and many rendering and texturing options including real-time physically based rendering and raytracing.
* [hdospray](https://github.com/ospray/hdospray) - The ospray for hydra is an open-source plugin for Pixar's USD to extend the hydra rendering framework with Intel Ospray. It is highly optimized for Intel CPU architectures ranging from laptops to large-scale distributed HPC systems.
* [LightWave Explorer](https://github.com/NickKarpowicz/LightwaveExplorer) - Lightwave explorer is an open source nonlinear optics simulator, intended to be fast, visual, and flexible for students and researchers to play with ultrashort laser pulses and nonlinear optics without having to buy a laser first.
* [oidn](https://github.com/OpenImageDenoise/oidn) - Intel Open Image Denoise is an open-source library for image denoising in ray tracing rendering applications with high quality and performance, thanks to efficient deep learning-based filters that can be trained using the included toolkit and user-provided image datasets.
* [openpgl](https://github.com/OpenPathGuidingLibrary/openpgl) - The Intel Open Path Guiding Library (Open PGL) implements path guiding into a renderer, offering implementations of current state-of-the-art path guiding methods which increase the sampling quality and renderer efficiency.
* [ospray](https://github.com/ospray/ospray) - Ospray is an open source, scalable and portable ray tracing engine designed for high fidelity visualization on Intel architecture CPUs. It allows users to easily build interactive applications using ray-tracing based rendering for both surface and volume-based visualizations.
* [ospray_studio](https://github.com/ospray/ospray_studio) - Ospray Studio is an open-source, interactive visualization and ray tracing application that utilizes Intel Ospray as its core rendering engine. Users can create scene graphs to render complex scenes with high-fidelity or very large scenes requiring supercomputing resources.
* [tracer](https://github.com/JoshuaSenouf/tracer) - Tracer is a renderer that uses Embree and USD to produce photorealistic images using path tracing on the CPU, with features like subpixel jitter antialiasing, depth of field, and a variety of integrators.
* [vistle](https://github.com/vistle/vistle) - Vistle is a modular data-parallel visualization system. It requires a C++14 compatible compiler that supports ISO/IEC 14882:2014, alongside compiling requirements of Boost, CMake and MPI. Additionally, it supports Covise, OpenCover, OpenSceneGraph and Qt 5 libraries, and also provides support code, rendering libraries, controlling code for Vistle session and visualization algorithm modules.

### Energy

* [A DPC++ Backend for the OCCA Portability Framework](https://github.com/libocca/occa) - OCCA—an open source portable and vendor neutral framework for parallel programming on heterogeneous platforms—is used by mission critical computational science and engineering applications of public and private sector organizations including the U.S. Department of Energy and Shell.

### Gaming

* [NovelRT](https://github.com/novelrt/NovelRT) - NovelRT is a cross-platform game engine for visual novels and 2D games. It is still in the early alpha stage, but currently supports graphics and audio.

### Manufacturing

* [S3_DeformFDM](https://github.com/zhangty019/S3_DeformFDM) - The S3 Slicer is a framework for achieving support-free strength reinforcement and surface quality in multi-axis 3D printing by computing the rotation-driven deformation for the input model.

### Misc

* [MuSYCL](https://github.com/keryell/muSYCL) - muSYCL, the SYCL musical! This is a small music synthesizer to experiment with C++23 programming, design patterns and acceleration on hardware accelerators like GPU, FPGA or CGRA with the SYCL 2020 standard.
* [SYCL-samples](https://github.com/codeplaysoftware/SYCL-samples) - A collection of samples written using the SYCL standard for C++.


### Mathematics and Science

* [1D Heat Transfer Simulation](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/StructuredGrids/1d_HeatTransfer) - (C++ based, from Intel) This 1D-Heat-Transfer sample is an application that simulates the heat propagation on a one-dimensional isotropic and homogeneous medium. The code sample includes both parallel and serial calculations of heat propagation.
* [3D Wave Simulation](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/StructuredGrids/guided_iso3dfd_GPUOptimization) - (C++ based, from Intel) The ISO3DFD sample refers to Three-Dimensional Finite-Difference Wave Propagation in Isotropic Media; it is a three-dimensional stencil to simulate a wave propagating in a 3D isotropic medium. Starts with a simple serial implementation and shows how to use SYCL to offload to the GPU. Then shows how to optimize.
* [ACTS GPU Ramp](https://github.com/acts-project/traccc) - Demonstrator tracking chain on accelerators
* [arpack-ng](https://github.com/opencollab/arpack-ng) - Arpack ng is a collection of Fortran77 subroutines designed to solve large scale eigenvalue problems and is a community project maintained by volunteers.
* [Amber](https://ambermd.org/GetAmber.php) Amber is a high-performance molecular dynamics (MD) code used by thousands of scientists in academia, national labs, and industry for computational drug discovery and related research.
* [ATLAS Charged Particle Seed Finding with DPC++](https://github.com/acts-project/acts) - The ATLAS Experiment is one of the general-purpose particle physics experiments built at the Large Hadron Collider (LHC) at CERN in Geneva. Its goal is to study the behavior of elementary particles at the highest energies ever produced in a laboratory help us better understand universe.
* [bfs-sycl-fpga](https://github.com/kaanolgu/bfs-sycl-fpga) - The Breadth-First Search algorithm implementations _memoryBFS_ and _streamingBFS_ using Intel oneAPI (SYCL2020) on Intel FPGAs
* [dedekind-MKL](https://github.com/stefan-zobel/dedekind-MKL) - Selected BLAS and LAPACK Java bindings for Intel's oneAPI Math Kernel Library (oneMKL) on Windows and Linux.
* [Discrete Cosine Transform Imeage Compression](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/SpectralMethods/DiscreteCosineTransform) - (C++ based, from Intel) The Discrete Cosine Transform (DCT) sample demonstrates how DCT and Quantizing stages can be implemented to run faster using SYCL* by offloading image processing work to a GPU or other device.
* [Direction Field Visualization with Python](https://github.com/olutosinbanjo/direction_field) - This project demonstrates the visualization of a direction field with Python using the differential equation of a falling object as a case study.  The effectiveness of Heterogeneous Computing is also shown by exploring optimized libraries added functionalities in Intel® Distribution for Python.
* [GinkgoOneAPI](https://github.com/ginkgo-project/ginkgo) - In this project we want to explore the potential of having an Intel OneAPI backend for the Gingko software package: https://ginkgo-project.github.io/
* [GROMACS](https://www.gromacs.org/) A free and open-source software suite for high-performance molecular dynamics and output analysis.
* [repulsive-surfaces](https://github.com/icethrush/repulsive-surfaces) - A numerical framework for optimization of surface geometry while avoiding (self-)collision.
* [Grid](https://github.com/dbollweg/Grid.git) - Data parallel C++ mathematical object library.
* [gtensor](https://github.com/wdmapp/gtensor) - gtensor is a multi-dimensional array C++14 header-only library for hybrid GPU development. It was inspired by xtensor, and designed to support the GPU port of the GENE fusion code.
* [Homogeneous and Heterogeneous Implementations of a tridiagonal solver on Intel® Xeon® E-2176G with oneMKL getrs](https://github.com/olutosinbanjo/oneMKL_getrs.git) - Homogeneous and Heterogeneous implementations of a tridiagonal solver with oneMKL getrs
* [Jacobi Iterative Solver for Multi-GPU](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/DenseLinearAlgebra/guided_jacobi_iterative_gpu_optimization) - (C++ based, from Intel) Illustrates how to use the Jacobi Iterative method to solve linear equations. This sample starts with a CPU-oriented application and shows how to use SYCL to offload regions of the code to a GPU. The sample walks through developing an optimization strategy by iteratively optimizing the code and ultimately targetting multi-GPUs if available.
* [LAMMPS](https://github.com/lammps/lammps) - LAMMPS is a classical molecular dynamics simulation code designed to run efficiently on parallel computers.  It was developed at Sandia National Laboratories, a US Department of Energy facility, with funding from the DOE.  It is an open-source code, distributed freely under the terms of the GNU Public License (GPL) version 2.
* [mapmap_cpu](https://github.com/dthuerck/mapmap_cpu) - MapMap CPU is a massively parallel generic MRF map solver with minimal input assumptions, capable of solving a large class of MRF problems.
* [MF-LBM](https://github.com/lanl/MF-LBM) - This is a lattice Boltzmann code designed for direct numerical simulation of flow in porous media. It is written in Fortran 90 and optimized for vectorization and parallel programming.
code to SYCL. 
* [Monte Carlo Based Finanical Simulation for Multi-GPU](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/MapReduce/guided_MonteCarloMultiGPU_SYCLMigration) - (C++ based, from Intel) Evaluates fair call price for a given set of European options using the Monte Carlo approach. MonteCarlo simulation is one of the most important algorithms in quantitative finance. This sample uses a single CPU Thread to control multiple GPUs. Shows how to migrate CUDA based code to SYCL.
* [mt-kahypar](https://github.com/kahypar/mt-kahypar) - MT-KaHyPar is a multi-threaded algorithm for partitioning graphs and hypergraphs. It aims to minimize an objective function defined on the hyperedges while balancing block sizes and optimizing connectivity. It can partition extremely large graphs and hypergraphs with comparable solution quality to the best sequential graph partitioners while being more than an order of magnitude faster with only ten threads.
* [NAMD](https://www.ks.uiuc.edu/Research/namd/) is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems.
* [NWGraph](https://github.com/pnnl/NWGraph) - The Northwest Graph Library (NWGraph) is a high-performance header-only generic C++ graph library based on C++20 concepts and ranges. It includes multiple graph algorithms for well-known graph kernels and supporting data structures.
* [octotiger](https://github.com/STEllAR-GROUP/octotiger) - Octo-Tiger is an astrophysics program simulating the evolution of star systems based on the fast multipole method on adaptive Octrees. It was implemented using high-level C++ libraries, specifically HPX and Vc, which allows its use on different hardware platforms.
* [Odd Even Merge and Sorting](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/StructuredGrids/1d_HeatTransfer) - (C++ based, from Intel) Demonstrates how to use the odd-even mergesort algorithm (also known as "Batcher's odd–even mergesort") which may benefit whenn working with  batches of short-sized to mid-sized (key, value) array pairs.  Shows how to migrate CUDA based code to SYCL.
* [Optical Flow Method](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/C%2B%2BSYCL/StructuredGrids/guided_HSOpticalflow_SYCLMigration) - (C++ based, from Intel) The HSOpticalFlow sample is a computation of per-pixel motion estimation between two consecutive image frames caused by movement of object or camera. Shows how to migrate CUDA based code to SYCL.
* [portBLAS](https://github.com/codeplaysoftware/portBLAS) - An implementation of BLAS using the SYCL open standard.
* [PyPardisoProject](https://github.com/haasad/PyPardisoProject) - Pypardiso is a Python package for solving large sparse linear systems of equations using the Intel oneAPI Math Kernel Library Pardiso solver. It provides the same functionality as Scipy's spsolve but is faster in many cases.
* [qmckl_sycl](https://github.com/elielnd/qmckl_sycl) - SYCL GPU port of the [QMCkl: Quantum Monte Carlo Kernel Library](https://github.com/TREX-CoE/qmckl). 
* [repulsive-surfaces](https://github.com/icethrush/repulsive-surfaces) - A numerical framework for optimization of surface geometry while avoiding (self-)collision.
* [SPHinxXsys](https://github.com/Xiangyu-Hu/SPHinXsys) - SPHinXsys provides C++ APIs for physically accurate simulation and optimization. It aims to handle coupled industrial dynamic systems including fluid, solid, multi-body dynamics and beyond. The multi-physics library is based a unique and unified computational framework by which strong couplings have been achieved for all involved physics.
[suanPan](https://github.com/TLCFEM/suanPan) - suanPan is a finite element method (FEM) simulation platform for applications in fields such as solid mechanics and civil/structural/seismic engineering. The name suanPan (in some places such as suffix it is also abbreviated as suPan) comes from the term Suan Pan (算盤), which is Chinese abacus.
[sycl-collision-sim](https://github.com/rafbiels/sycl-collision-sim) - Demo 3D simulation of rigid body physics with different shapes bouncing off each other confined in a box. Two implementations are provided, one sequential with standard C++ code compiled for CPU, and parallel SYCL implementation which can be compiled for any target device (e.g. a GPU) supported by a SYCL compiler.

### Tools and Development
* [ArrayFire - oneAPI Backend](https://github.com/arrayfire/arrayfire) - ArrayFire is a general-purpose tensor library that simplifies the process of software development for the parallel architectures found in CPUs GPUs and other hardware acceleration devices. This project is to develop a oneAPI backend to the library which currently supports CUDA OpenCL and x86.
* [ArrayFire - Rust Bindings](https://github.com/arrayfire/arrayfire-rust) - Rust bindings for ArrayFire  a general-purpose tensor library that simplifies the process of software development for the parallel architectures found in CPUs GPUs and other hardware acceleration devices. This project is to develop a oneAPI backend to the library which currently supports CUDA OpenCL and x86.
* [amrex-sycl](https://github.com/farscape-project/amrex-sycl) - A SYCL plug-in to run AMReX apps on AMD/Nvidia GPUs. The plug-in consists of a build script and code patches which extend AMReX's SYCL capability beyond Intel GPUs.
* [chip-spv](https://github.com/CHIP-SPV/chip-spv) - The "chip spv" project allows for the portability of HIP and CUDA applications to platforms supporting SPIR-V. Currently, it offers support for OpenCL and Level-Zero as low-level runtime alternatives.
Selected BLAS and LAPACK Java bindings for Intel's oneAPI Math Kernel Library on Windows and Linux
* [dedekind-MKL](https://github.com/stefan-zobel/dedekind-MKL) - Selected BLAS and LAPACK Java bindings for Intel's oneAPI Math Kernel Library (oneMKL) on Windows and Linux.
* [dpctl](https://github.com/IntelPython/dpctl) - Python SYCL bindings and SYCL-based Python Array API library.
* [formulog](https://github.com/HarvardPL/formulog)** - Formulog is a logic programming language that supports Datalog, SMT queries, and first-order functional programming. It requires JRE 11 and a supported SMT solver, such as Z3, Boolector, CVC4, or Yices.
* [HeCBench](https://github.com/zjin-lcf/HeCBench) - The hecbench repository contains a collection of benchmarks for studying performance portability and productivity with various heterogeneous computing languages.The benchmarks are divided into categories like computer vision, bioinformatics, and finance.
* [HPCToolKit](http://hpctoolkit.org/) - HPCToolkit is an open-source performance tool that is in some respects similar to VTune though it also works on Power and ARM architectures. It also works on NVIDIA and AMD GPUs. Our aim is to also use it for performance analysis of Intel GPUs with Intel’s OpenCL to our targets as a prelude to A0
* [kharma](https://github.com/AFD-Illinois/kharma) - Kokkos-based High-Accuracy Relativistic Magnetohydrodynamics with AMR. KHARMA is an implementation of the HARM scheme for gerneral relativistic magnetohydrodynamics (GRMHD) in C++. It is based on the Parthenon AMR infrastructure, using Kokkos for parallelism and GPU support.
* [Kokkos](https://github.com/kokkos/kokkos) - Kokkos Core implements a programming model in C++ for writing performance portable applications targeting all major HPC platforms. For that purpose it provides abstractions for both parallel execution of code and data management. Kokkos is designed to target complex node architectures with N-level memory hierarchies and multiple types of execution resources. It currently can use CUDA, HIP, SYCL, HPX, OpenMP and C++ threads as backend programming models with several other backends in development.
* [levelzero-jni](https://github.com/beehive-lab/levelzero-jni) -  Intel LevelZero JNI library for TornadoVM. This project is a Java Native Interface (JNI) binding for Intel's Level Zero. This library is as designed to be as closed as possible to the LevelZero API for C++.
* [libxsmm](https://github.com/libxsmm/libxsmm) - LIBXSMM is a library for specialized dense and sparse matrix operations as well as for deep learning primitives such as small convolutions. 
* [mixbench](https://github.com/ekondis/mixbench) - A GPU benchmark tool for evaluating GPUs and CPUs on mixed operational intensity kernels (CUDA, OpenCL, HIP, SYCL, OpenMP)
* [numba-dpex](https://github.com/IntelPython/numba-dpex) - Numba dpex is an extension for the Numba Python JIT compiler that provides a kernel programming API and an offload feature. It supports devices including Intel CPUs, integrated GPUs, and discrete GPUs.
* [oneapi-asp](https://github.com/OFS/oneapi-asp) - Intel® oneAPI Accelerator Support Package (ASP) for Open FPGA Stack (OFS)
* [oneapi-containers](https://github.com/intel/oneapi-containers) - The Intel OneAPI Containers simplify programming by delivering the tools to deploy applications and solutions on various architectures. These containers allow developers to set up and distribute environments for profiling and execute applications built with OneAPI toolkits.
* [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) - The oneapi.jl GitHub project provides support for working with the oneapi unified programming model and offers low-level wrappers for the level zero library, kernel programming, and high-level array programming capabilities.
* [Open-source Scientific Applications and Benchmarks](https://github.com/zjin-lcf/oneAPI-DirectProgramming) - This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA, SYCL, and OpenMP target offloading. Intel® DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program.
* [p2rng](https://github.com/arminms/p2rng) -  A modern header-only C++ library for parallel algorithmic (pseudo) random number generation supporting OpenMP, CUDA, ROCm and oneAPI.
* [PTXprofiler](https://github.com/ProjectPhysX/PTXprofiler) - A simple profiler to count Nvidia PTX assembly instructions of OpenCL/SYCL/CUDA kernels for roofline model analysis.
* [PySYCL](https://gthub.com/PySYCL/PySYCL) - SYCL functionalities within Python for GPU targeted development.
* [RayBNN_Raytrace](https://github.com/BrosnanYuen/RayBNN_Raytrace) -  Ray tracing library using GPUs, CPUs, and FPGAs via CUDA, OpenCL, and oneAPI 
* [RcppParallel](https://github.com/RcppCore/RcppParallel) - The rcppparallel project provides high-level functions for parallel programming with Rcpp and supports using Intel TBB for performance on Windows, macOS, and Linux systems. 
* [R-oneMKL](https://github.com/R-OneMKL/oneMKL) - The oneMKL package establishes the connection between the R environment and Intel oneAPI Math Kernel Library (oneMKL), a prerequisite of using oneMKL.MatrixCal package. Specifically, oneMKL provides necessary header files and dynamic library files to R, and imports files from the packages mkl, mkl-include, and intel-openmp from Anaconda. 
* [SimSYCL](https://github.com/celerity/SimSYCL) - SimSYCL is a single-threaded, synchronous, library-only implementation of the SYCL 2020 specification. It enables you to test your SYCL applications against simulated hardware of different characteristics and discover bugs with its extensive verification capabilities.
* [Spyker](https://github.com/ShahriarRezghi/Spyker) - High-performance Spiking Neural Networks Library Written From Scratch with C++ and Python Interfaces.
* [SYCLomatic](https://github.com/oneapi-src/SYCLomatic) - The SycloMatic project helps developers migrate code to the SYCL heterogeneous programming model. Daily builds are available, but not rigorously tested for production quality control.
* [SYnergy](https://github.com/unisa-hpc/SYnergy) -  Energy Measurement and Frequency Scaling for SYCL applications. 
* [SYCLops](https://github.com/Huawei-PTLab/SYCLops) - A SYCL-specific LLVM-to-MLIR converter.
* [syclreduce](https://github.com/ORNL/syclreduce) - This is a tiny package implementing what is a giant unmet need in SYCL2020 - proper reductions. Want to sum a vector coming from every thread in a kernel launch? Want to accumulate a couple different kinds of diagnostic output from a kernel? Too bad. SYCL doesn't have full documentation on how span<> works, and you'll easily get lost writing your own undefined type reducer.
* [TAU Performance System](https://github.com/UO-OACISS/tau2) - The TAU Performance System® supports profiling and tracing of programs written using the Intel OneAPI. Intel OneAPI provides two interfaces for programming - OpenCL and DPC++/SYCL for CPUs and GPUs. TAU supports both - the OpenCL profiling interface and Intel Level Zero API to observe performance. 
* [TornadoVM](https://github.com/beehive-lab/TornadoVM) - TornadoVM is an open-source software technology that automatically accelerates Java programs on multi-core CPUs GPUs and FPGAs.
* [toyBrot](https://gitlab.com/VileLasagna/toyBrot) - toyBrot is a raymarching fractal generator that is used both as a  simple benchmarking tool and a study tool for parallelisation. The code is is implemented with over 10 different technologies including Intel TBB ISPC and SYCL (with support for oneAPI)
* [XFluids](https://github.com/XFluids/XFluids) - a unified cross-architecture heterogeneous CFD solver that suports Nvidia, Amd and Intel GPUs.
* [ZFP](https://github.com/LLNL/zfp) - zfp is a compressed format for representing multidimensional floating-point and integer arrays. zfp provides compressed-array classes that support high throughput read and write random access to individual array elements. zfp also supports serial and parallel compression of whole arrays for applications that read and write large data sets to and from disk.

## Tutorials

* [50YearsOfRayTracing](https://github.com/neil3d/50YearsOfRayTracing) - This GitHub project is focused on ray tracing and covers several techniques and models developed from 1968 to 1997, with a focus on physically based rendering.
* [data-parallel-CPP](https://github.com/Apress/data-parallel-CPP) - The Data Parallel C Book Source Samples repository contains code that accompanies the Data Parallel C: Mastering DPC for Programming of Heterogeneous Systems using C++ and SYCL book.
* [efficient-dl-systems](https://github.com/mryab/efficient-dl-systems) - This repository contains materials for the Efficient Deep Learning Systems course taught at the HSE University and Yandex School of Data Analysis.
* [Jurassic](https://github.com/IntelSoftware/Jurassic) - Hunting Dinosaur bones using AI
* [syclacademy](https://github.com/codeplaysoftware/syclacademy) - SYCL Academy, a set of learning materials for SYCL heterogeneous programming
