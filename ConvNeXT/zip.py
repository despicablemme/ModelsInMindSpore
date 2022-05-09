import zipfile

path = '/home/work/user-job-dir/data/'

# path = 'G:\项目\ConvNeXt-main.zip'
if zipfile.is_zipfile(path):
    print(123)
    print("====================================================")
    f = zipfile.ZipFile(path)
    files = f.namelist()
    print(files)
    print("====================================================")
else:
    print(321)
