# Support for large analysis files, up to 2Gb, using mmap'd arrays



_MEM_CHUNK = 64000 # size of a working chunk of memory
_D_TYPE = 'float32' # data type for arrays

def open_mmapped_array_readonly(filename, shape, dtype=_D_TYPE):
    return np.memmap(filename, dtype=dtype, mode='r', shape=shape)

def open_mmapped_array_readwrite(filename, shape, dtype=_D_TYPE):
    return np.memmap(filename, dtype=dtype, mode='r+', shape=shape)

def new_mmapped_array(filename, shape, dtype=_D_TYPE):
    return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)


def process_array(fp):
    r,c = fp.shape
    mem_rows =_MEM_CHUNK / c
    for m in arange(0, r, mem_rows):
        print fp[m:m+mem_rows,:]

def foo(filename='/tmp/foo', test_shape=(100000,128) ):
    fp1 = new_mmapped_array(filename, shape=test_shape)
    fp2 = open_mmapped_array_readonly(filename, shape=test_shape)
    return fp1, fp2

