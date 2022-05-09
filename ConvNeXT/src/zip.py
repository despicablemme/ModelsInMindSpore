import zipfile

path='/home/work/user-job-dir/data/imagenet.zip'
# path = 'G:\项目\ConvNeXt-main.zip'
if zipfile.is_zipfile(path):
    print(123)
    f = zipfile.ZipFile(path)
    files = f.namelist()
    print(files)
else :
    print(321)