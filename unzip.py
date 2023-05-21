import zipfile

f = zipfile.ZipFile("./CLEVR_v1.0.zip",'r') # 压缩文件位置
for file in f.namelist():
    f.extract(file,"/nobackup/users/zitian/code/Heaplax/clevr/")               # 解压位置
f.close()