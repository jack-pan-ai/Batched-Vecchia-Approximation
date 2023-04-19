# Vecchia-kblas-gpu

What is KBLAS
=============

KAUST BLAS (KBLAS) is a high performance CUDA library implementing a subset of BLAS as well as Linear Algebra PACKage (LAPACK) routines on NVIDIA GPUs. Using recursive and batch algorithms, KBLAS maximizes the GPU bandwidth, reuses locally cached data and increases device occupancy. KBLAS supports operations for regular dense and hierarchical low-rank matrix workloads. KBLAS provides, therefore, the critical building blocks not only for the traditional high performance dense linear algebra libraries, but also for the emerging numerical libraries supporting hierarchical low-rank matrix computations. This is a major leap toward leveraging GPU hardware accelerators for high performance low-rank matrix approximations and computations, which currently drives the research agenda of the linear algebra scientific community.

KBLAS is written in CUDA C. It requires CUDA Toolkit for installation.

What is Vecchia
===============
Vecchia approximation is a statistical method used to approximate complex spatial models. It is a technique for approximating the covariance matrix of a spatial random field based on a subset of the available data. The approach was first introduced by Daniela Cocchi and Renato L. Briganti in 2011, and it has since been widely used in spatial statistics.
