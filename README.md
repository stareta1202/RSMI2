# RSMI




##  How to use

### 1. Required libraries

#### LibTorch
homepage: https://pytorch.org/get-started/locally/

CPU version: https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip

For GPU version, you need to choose according to your setup.

#### boost

homepage: https://www.boost.org/
Download [boost 1.75.0](https://www.boost.org/doc/libs/1_75_0/more/getting_started/unix-variants.html) and unzip the file. 
Relocate the unzipped files to the IncludePath. 

#### 2. Change Makefile

Choose CPU or GPU

```
# TYPE = CPU
TYPE = GPU

Change *home/liuguanli/Documents/libtorch_gpu* to your own path.

ifeq ($(TYPE), GPU)
	INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
else
	INCLUDE = -I/home/liuguanli/Documents/libtorch/include -I/home/liuguanli/Documents/libtorch/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch/lib
endif
```
#### 3. Change Exp.cpp

comment *#define use_gpu* to use CPU version

```C++
#ifndef use_gpu
#define use_gpu
.
.
.
#endif  // use_gpu
```

#### 4. Run

```bash
make clean
make -f Makefile
./Exp <data directory> <query directory>
# OR
./Exp sequential <query directory>
```