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
4. [AI - Data Science](#AI-\-Data-Science)
5. [Medical and Life Sciences](#Medical-and-Life-Sciences)
6. [Mathematics and Science](#Mathematics-and-Science)
7. [Security](#Security)
8. [Autonomous Systems](#Autonomous-Systems)
9. [Tools & Development](#Tools-and-Development)
10. [Energy](#Energy)
<!-- 11. [Financial Services](#Financial-Services)
12. [Manufacturing](#Manufacturing)
13. [Tutorials](#Tutorials) -->


## Projects

### AI - Machine Learning
* [Performance and Portability Evaluation of the K-Means Algorithm on SYCL with CPU-GPU architectures](https://github.com/artecs-group/k-means) - This work uses the k-means algorithm to asses the performance portability of one of the most advanced implementations of the literature He-Vialle over different programming models (DPC++ CUDA OpenMP) and multi-vendor CPU-GPU architectures.
* [DQRM](https://anonymous.4open.science/r/Deep_Quantized_Recommendation_Model_DQRM-6B4D) - Deep Quantized Recommendation Model (DQRM) is a recommendation framework that is small, powerful in inference, and efficient to train.

### AI - Natural Language Processing
* [Gavin AI](https://github.com/Gavin-Development/GavinTraining) - Gavin AI is a project created by Scot_Survivor (Joshua Shiells) ShmarvDogg which aims to have English human like conversations through the use of AI and ML. Gavin works on the Transformer architecture however Performer FNet architectures are being investigated for better scaling.
* [Language Identification](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads/LanguageIdentification) (Python based) Trains a model to perform language identification using the Hugging Face Speechbrain library and CommonVoice dataset, and optimized with IPEX and INC.
* [Census](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads/Census)  (Python based) Use Intel® Distribution of Modin to ingest and process U.S. census data from 1970 to 2010 in order to build a ridge regression based model to find the relation between education and the total income earned in the US.


### AI - Computer Vision
* [LidarObjectDetection-PointPillars](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads/LidarObjectDetection-PointPillars) (C++ based, requires AI toolkit and OpenVINO). demonstrates how to perform 3D object detection and classification using input data (point cloud) from a LIDAR sensor.

* [Certiface Anti-Spoofing](https://github.com/cabelo/oneapi-antispoofing) - Certiface AntiSpoofing use oneAPI for fast decode video for perform liveness detection with inference. The system is capable of spotting fake faces and performing anti-face spoofing in face recognition systems.
* [LidarObjectDetection-PointPillars]() (C++ based, requires AI toolkit and OpenVINO). demonstrates how to perform 3D object detection and classification using input data (point cloud) from a LIDAR sensor.

<!-- ### AI - NL -->

### AI - Toolkits

* [Invoice To Cash Automation](https://github.com/oneapi-src/invoice-to-cash-automation) - AI toolkit to extract information from claim documents to categorize the claims. Helps develop models to accelerate the resolution of accounts receivable claims for trade promotion deductions.

* [AI Personal Identifiable Information Data Protection](https://github.com/oneapi-src/ai-data-protection) - Provides anonyimzation functions, which include methods for masking, hashing and encrypting/decrypting the PII data in large datasets. Can be used to protect the privacy and security of individuals in a dataset.

* [AI Structured Data Generation](https://github.com/oneapi-src/ai-structured-data-generation) - Generate structured synthetic data for training and inferencing.
* [Data Streaming Anomaly Detection](https://github.com/oneapi-src/data-streaming-anomaly-detection) - help detect anomalies using tensorflow and oneAPI to build a deep learning model that can detect anomalies in data collected from a IOT device to monitor equipment condition and prevent any issue from being cascaded.
* [Intelligent Indexing](https://github.com/oneapi-src/intelligent-indexing) - a reference kit to build an AI-based Natural Language Processing solution for classifying documents.
* [Network Intrusion Detection](https://github.com/oneapi-src/network-intrusion-detection) - A pattern based network intrusion system using oneAPI and machine learning.
* [Customer Chatbot](https://github.com/oneapi-src/customer-chatbot) - a pytorch based conversational AI chatbot for customer care.
* [Strutucal Damage Assessment](https://github.com/oneapi-src/structural-damage-assessment) - A PyTorch-based AI model that works on satellite-captured images to assess the severity of damage in the aftermath of a natural disaster.
* [Text Data Generation](https://github.com/oneapi-src/text-data-generation) - Creates synthetic data that is artificially generated. This reference kit uses a pre-trained GPT2 modle provided by hugging face to generate synthetic data applicable to product testing and training machine learning algorithms without running into privacy issues.
* [Traffic Camera Object Detection](https://github.com/oneapi-src/traffic-camera-object-detection) - reference kit demonstrating how to improve traffic using a number of different technology and oneAPI.
* [Vertical Search Engine](Semantic Vertical Search Engine) - Demonstrates a possible reference implementation of a deep learning based NLP pipeline for semantic search of an organization's document using a pre-trained model.
* [Visual Process Discovery](https://github.com/oneapi-src/visual-process-discovery) - A reference kit implementing visual process discovery. VPDs can be used to enhance customer experience by providing personalized solutions knowing their needs as they navigate through a company's website.
* [Synthetic Voice/Audio Generation](https://github.com/oneapi-src/voice-data-generation) - Generate synthetic voices and speeches - can be used in chatbots, virtual assistants, and is applicable in a host of applications. Voice synthesis technology is increasingly used to create more natural sounding virtual assistants.
* [Predictive Asset Maintenance](https://github.com/oneapi-src/predictive-asset-health-analytics) - Shows an alternative method of using oneAPI AI Analytics Toolkit over the stock version of the same package like XGBoost.
* [Historical Assets Document Processing \(OCR\)](https://github.com/oneapi-src/historical-assets-document-process) - Allows you to process large amounts of structured, semi-structured and unstructured content in documents. Through the use of image processing, analysis, text region detection and text extraction using OCR - the results can then be stored and can be put into a database.
* [Drone Navigation Inspection](https://github.com/oneapi-src/drone-navigation-inspection) - Find safe drone landing zone without damaging property or injuring people using oneAPI and TensorFlow.
* [Power Line Fault Detection](https://github.com/oneapi-src/powerline-fault-detection) - Process and analyze signals from a 3-phase power supply system used in power lines to predict whether or not a signal has a partial discharge using SciPy and NumPy calculations.
* [Loan Default Risk Prediction](https://github.com/oneapi-src/loan-default-risk-prediction) - Train and utilize an AI model using XGBoost to predict the probability of a loan default from client characteristics and the type of loan obligation.
* [Credit Card Fraud Detection](https://github.com/oneapi-src/credit-card-fraud-detection) - Uses Intel AI Analytics Toolkit and scikit-learn to train a AI algorithm to detect credit card fraud.
* [Visual Quality Inspection](https://github.com/oneapi-src/visual-quality-inspection) - Build a computer vision based model for building quality visual inspection based on a dataset from the pharma industry.
* [Disease Prediction](https://github.com/oneapi-src/disease-prediction) - Demonstrates using a deep learning based NLP pipeline to train a document classifier that takes in notes from patient's symptoms and predicts teh diagnoses among a set of known diseases.
* [Documentation Automation](https://github.com/oneapi-src/document-automation) - based on the Tensorflow BERT transfer learning NER Model, build a deep learning model to predict the named entity tags for a given sentence.
* [AI based transcribing](https://github.com/oneapi-src/ai-transcribe) - A reference solution showing how to use speech to text conversion to convert audio session tapes into digital notes in a psychologist's office.
* [Image Data Generation](https://github.com/oneapi-src/image-data-generation) - An AI-enabled image generator that aids in generating accurate image and image segmentation datasets where availability of such datasets are limited.
* [Medical Imaging Diagnostics](https://github.com/oneapi-src/medical-imaging-diagnostics) - Using machine learning and deep learning, train an AI algorithm that identifies images that warrant further attention to classify abnormalities.
* [Engineering Design Optimizations](https://github.com/oneapi-src/engineering-design-optimization) - Train a model to create new bicycle designs with unique frames and handles, and generalize rare novelties to a broad set of designs, competely automatic and without requiring human intervention.
* [Digital Twin for Design Exploration](https://github.com/oneapi-src/digital-twin) - A model that can be used to test digital replicas of real world products or devices for faults.
* [Purchase Prediction](https://github.com/oneapi-src/purchase-prediction) - A oneAPI based reference AI model that uses machine learning to predict purchases of customers. 
* [Demand Forecasting](https://github.com/oneapi-src/demand-forecasting) - Builds and trains an AI model using deep learning to train and utiliez a CNN-LSTM time series model that predicts the next days demand every item based on 130 days worth of sales data.
* [Order to Delivery Time Forecasting](https://github.com/oneapi-src/order-to-delivery-time-forecasting) - A machine learning based predictive model that provides delivery time forecasting for e-commerce platform.
* [Customer Segmentation for Online Retailers](https://github.com/oneapi-src/customer-segmentation) - Demonstrates how machine learning can aid in building a deeper understanding of a businesses clientele by segmenting customers into clusters that can be used to implement personalized and targeted campaign.
* [Product Recommedation](https://github.com/oneapi-src/product-recommendations) - A reference kit that demonstrates one way where AI can be used to build a recommendation system for an e-commerce business using scikit-learn and oneAPI.
* [Customer Churn Prediction](https://github.com/oneapi-src/customer-churn-prediction) - Using historical customer churn data along with service details, a machine learning model built to predict whether the customer is going to churn. Reducing churn is key in the telecommunications industry to attract new customers and avoid contract terminations.




### AI - Data Science

* [fastRAG](https://github.com/IntelLabs/fastRAG) - Build and explore efficient retrieval-augmented generative models and applications. It's main goal is to make retrieval augmented generation as efficient as possible through the use of state-of-the-art and efficient retrieval and generative models.

* [HIAS TassAI Facial Recognition Agent](https://github.com/AIIAL/HIAS-TassAI-Facial-Recognition-Agent) - Security is an important issue for hospitals and medical centers to consider. Today's Facial Recognition can provide ways of automating security in the medical industry reducing staffing costs and making medical facilities safer for both patients and staff.

* [Drift Detection for Edge IoT Applications](https://github.com/blackout-ai/Face_Aging_Concept_Drift) - This concept drift project is run on video and image datasets such that we can calculate an overall precision and standard error. The concept drift detection technique finds True positives and False negatives using real and virtual drift detection. 

* [Boosting epistasis detection on Intel CPU+GPU systems](https://github.com/hiperbio/cross-dpc-episdet) - This work focuses on exploring the architecture of Intel CPUs and Integrated Graphics and their heterogeneous computing potential to boost performance and energy-efficiency of epistasis detection. This will be achieved making use of OpenCL Data Parallel C++ and OpenMP programming models.

### Mathematics and Science
* [GROMACS](https://www.gromacs.org/) A free and open-source software suite for high-performance molecular dynamics and output analysis.
* [GinkgoOneAPI](https://github.com/ginkgo-project/ginkgo) - In this project we want to explore the potential of having an Intel OneAPI backend for the Gingko software package: https://ginkgo-project.github.io/
* [ACTS GPU Ramp](https://github.com/acts-project/traccc) - D Projects
* [ATLAS Charged Particle Seed Finding with DPC++](https://github.com/acts-project/acts) - The ATLAS Experiment is one of the general-purpose particle physics experiments built at the Large Hadron Collider (LHC) at CERN in Geneva. Its goal is to study the behavior of elementary particles at the highest energies ever produced in a laboratory help us better understand universe.
* [Homogeneous and Heterogeneous Implementations of a tridiagonal solver on Intel® Xeon® E-2176G with oneMKL getrs](https://github.com/olutosinbanjo/oneMKL_getrs.git) - Homogeneous and Heterogeneous implementations of a tridiagonal solver with oneMKL getrs 
* [Direction Field Visualization with Python](https://github.com/olutosinbanjo/direction_field) - This project demonstrates the visualization of a direction field with Python using the differential equation of a falling object as a case study.  The effectiveness of Heterogeneous Computing is also shown by exploring optimized libraries added functionalities in Intel® Distribution for Python.
* [NAMD](https://www.ks.uiuc.edu/Research/namd/) is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems.
* [Amber](https://ambermd.org/GetAmber.php) Amber is a high-performance molecular dynamics (MD) code used by thousands of scientists in academia, national labs, and industry for computational drug discovery and related research.

### Autonomous Systems
* [Alice](https://github.com/intel/dffml/tree/alice/entities/alice/) - We are writing a tutorial for an open source project on how we build an AI to work on the open source project as if she were a remote developer. Bit of a self fulfilling prophecy but who doesn't love an infinite loop now and again.

### Data Visualization and Rendering

* [Substrate](https://github.com/seelabutk/substrate) - A toolset to help developers create and deploy cloud-based VaaS services (Visualization as a Service). Deployment targets include any platforms capable of running Docker Swarm, such as Amazon AWS, institutional clusters and even personal servers. Native for Python environment (pip installable).

### Tools and Development
* [ArrayFire - oneAPI Backend](https://github.com/arrayfire/arrayfire) - ArrayFire is a general-purpose tensor library that simplifies the process of software development for the parallel architectures found in CPUs GPUs and other hardware acceleration devices. This project is to develop a oneAPI backend to the library which currently supports CUDA OpenCL and x86.
* [Open-source Scientific Applications and Benchmarks](https://github.com/zjin-lcf/oneAPI-DirectProgramming) - This repository contains a collection of data-parallel programs for evaluating oneAPI direct programming. Each program is written with CUDA, SYCL, and OpenMP target offloading. Intel® DPC++ Compatibility Tool (DPCT) can convert a CUDA program to a SYCL program.
* [Scal](https://devmesh.intel.com/projects/scalsale) - New physical scalable benchmark, namely ScalSALE, is based on the well-known SALE scheme. ScalSALE's main goal is to provide a gold-standard benchmark application that can incorporate multi-physical schemes while maintaining scalable and efficient execution times.

* [TAU Performance System](https://github.com/UO-OACISS/tau2) - The TAU Performance System® supports profiling and tracing of programs written using the Intel OneAPI. Intel OneAPI provides two interfaces for programming - OpenCL and DPC++/SYCL for CPUs and GPUs. TAU supports both - the OpenCL profiling interface and Intel Level Zero API to observe performance. 
* [TornadoVM](https://github.com/beehive-lab/TornadoVM) - TornadoVM is an open-source software technology that automatically accelerates Java programs on multi-core CPUs GPUs and FPGAs.
* [toyBrot](https://gitlab.com/VileLasagna/toyBrot) - toyBrot is a raymarching fractal generator that is used both as a  simple benchmarking tool and a study tool for parallelisation. The code is is implemented with over 10 different technologies including Intel TBB ISPC and SYCL (with support for oneAPI)
* [HPCToolKit](http://hpctoolkit.org/) - HPCToolkit is an open-source performance tool that is in some respects similar to VTune though it also works on Power and ARM architectures. It also works on NVIDIA and AMD GPUs. Our aim is to also use it for performance analysis of Intel GPUs with Intel’s OpenCL to our targets as a prelude to A0
* [ZFP](https://github.com/LLNL/zfp) - zfp is a compressed format for representing multidimensional floating-point and integer arrays. zfp provides compressed-array classes that support high throughput read and write random access to individual array elements. zfp also supports serial and parallel compression of whole arrays for applications that read and write large data sets to and from disk.

### Energy

* [A DPC++ Backend for the OCCA Portability Framework](https://github.com/libocca/occa) - OCCA—an open source portable and vendor neutral framework for parallel programming on heterogeneous platforms—is used by mission critical computational science and engineering applications of public and private sector organizations including the U.S. Department of Energy and Shell.

<!-- ## Financial Services -->

<!-- ## Manufacturing -->
