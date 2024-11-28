import os
import loguru
from minio import Minio
from minio.error import S3Error
from datetime import timedelta
from minio.deleteobjects import DeleteObject
from tqdm import trange

class MinioConfig():
    endpoint = '127.0.0.1:9001'
    access_key = 'minioadmin'
    secret_key = 'minioadmin'
    secure = False


class Bucket(object):
    client = None
    policy = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetBucketLocation","s3:ListBucket"],"Resource":["arn:aws:s3:::%s"]},{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::%s/*"]}]}'
    def __new__(cls, *args, **kwargs):
        if not cls.client:
            cls.client = object.__new__(cls)
        return cls.client
    def __init__(self, secure=False):
        self.service = MinioConfig.endpoint
        self.client = Minio(MinioConfig.endpoint, MinioConfig.access_key, MinioConfig.secret_key, secure=secure)
    def exists_bucket(self, bucket_name):
        """
        判断桶是否存在
        :param bucket_name: 桶名称
        :return:
        """
        return self.client.bucket_exists(bucket_name=bucket_name)
    def create_bucket(self, bucket_name:str, is_policy:bool=True):
        """
        创建桶 + 赋予策略
        :param bucket_name: 桶名
        :param is_policy: 策略
        :return:
        """
        if self.exists_bucket(bucket_name=bucket_name):
            return False
        else:
            self.client.make_bucket(bucket_name = bucket_name)
        if is_policy:
            policy = self.policy % (bucket_name, bucket_name)
            self.client.set_bucket_policy(bucket_name=bucket_name, policy=policy)
        return True

    def get_bucket_list(self):
        """
        列出存储桶
        :return:
        """
        buckets = self.client.list_buckets()
        bucket_list = []
        for bucket in buckets:
            bucket_list.append(
                {"bucket_name": bucket.name, "create_time": bucket.creation_date}
            )
        return bucket_list

    def remove_bucket(self, bucket_name):
        """
        删除桶
        :param bucket_name:
        :return:
        """
        try:
            self.client.remove_bucket(bucket_name=bucket_name)
        except S3Error as e:
            print("[error]:", e)
            return False
        return True
    def bucket_list_files(self, bucket_name, prefix):
        """
        列出存储桶中所有对象
        :param bucket_name: 同名
        :param prefix: 前缀
        :return:
        """
        try:
            files_list = self.client.list_objects(bucket_name=bucket_name, prefix=prefix, recursive=True)
            for obj in files_list:
                print(obj.bucket_name,
                      # obj.object_name.encode('utf-8'),
                      obj.object_name,
                      obj.last_modified,
                      obj.etag, obj.size, obj.content_type)

            return files_list
        except S3Error as e:
            print("[error]:", e)
    def bucket_policy(self, bucket_name):
        """
        列出桶存储策略
        :param bucket_name:
        :return:
        """
        try:
            policy = self.client.get_bucket_policy(bucket_name)
        except S3Error as e:
            print("[error]:", e)
            return None
        return policy

    def download_file(self, bucket_name, file, file_path, stream=1024*32):
        """
        从bucket 下载文件 + 写入指定文件
        :return:
        """
        try:
            data = self.client.get_object(bucket_name, file)
            with open(file_path, "wb") as fp:
                for d in data.stream(stream):
                    fp.write(d)
        except S3Error as e:
            print("[error]:", e)
    def fget_file(self, bucket_name, file, file_path):
        """
        下载保存文件保存本地
        :param bucket_name:
        :param file:
        :param file_path:
        :return:
        """
        self.client.fget_object(bucket_name, file, file_path)
    def copy_file(self, bucket_name, file, file_path):
        """
        拷贝文件（最大支持5GB）
        :param bucket_name:
        :param file:
        :param file_path:
        :return:
        """
        self.client.copy_object(bucket_name, file, file_path)
    def upload_file(self,bucket_name, file, file_path, content_type):
        """
        上传文件 + 写入
        :param bucket_name: 桶名
        :param file: 文件名
        :param file_path: 本地文件路径
        :param content_type: 文件类型
        :return:
        """
        try:
            with open(file_path, "rb") as file_data:
                file_stat = os.stat(file_path)
                self.client.put_object(bucket_name, file, file_data, file_stat.st_size, content_type=content_type)
        except S3Error as e:
            print("[error]:", e)
    def fput_file(self, bucket_name, file, file_path):
        """
        上传文件
        :param bucket_name: 桶名
        :param file: 文件名
        :param file_path: 本地文件路径
        :return:
        """
        try:
            self.client.fput_object(bucket_name, file, file_path)
        except S3Error as e:
            print("[error]:", e)

    def stat_object(self, bucket_name, file):
        """
        获取文件元数据
        :param bucket_name:
        :param file:
        :return:
        """
        try:
            data = self.client.stat_object(bucket_name, file)
            print(data.bucket_name)
            print(data.object_name)
            print(data.last_modified)
            print(data.etag)
            print(data.size)
            print(data.metadata)
            print(data.content_type)
        except S3Error as e:
            print("[error]:", e)
    def remove_file(self, bucket_name, file):
        """
        移除单个文件
        :return:
        """
        self.client.remove_object(bucket_name, file)
    def remove_files(self, bucket_name, file_list):
        """
        删除多个文件
        :return:
        """
        delete_object_list = [DeleteObject(file) for file in file_list]
        for del_err in self.client.remove_objects(bucket_name, delete_object_list):
            print("del_err", del_err)
    def presigned_get_file(self, bucket_name, file, days=7):
        """
        生成一个http GET操作 签证URL
        :return:
        """
        return self.client.presigned_get_object(bucket_name, file, expires=timedelta(days=days))

    def Upload_folder(self, bucket_name, folder):
        """
        功能描述：将本地文件（夹），传输到MinIO指定Buckets
        PS1:支持断点续传
        公众号：CV初学者

        :param bucket_name: MinIO的Buckets名称
        :param folder: 本地待传文件夹地址
        :param client: 用户信息
        :return:
        """
        for root, dirs, files in os.walk(folder, topdown=False, followlinks=False):
            for i in trange(len(files)):
                filename = files[i]
                object_name = os.path.join(root, filename)[2:].replace('\\', '/')
                file_path = os.path.join(root, filename)
                try:
                    # 通过文件上传到对象中 fput_object(存储桶名称, 对象名称, 本地文件的路径)
                    self.client.fput_object(bucket_name=bucket_name, object_name="vector_store/"+filename, file_path=file_path)
                except Exception as e:
                    loguru.logger.error(e)
                    print('Upload error')
                # try:
                #     # 获取对象的元数据，能成功，就说明对象存在，否则对象出错，对象不存在
                #     self.client.stat_object(bucket_name=bucket_name, object_name=object_name)
                # except:
                #     try:
                #         # 通过文件上传到对象中 fput_object(存储桶名称, 对象名称, 本地文件的路径)
                #         self.client.fput_object(bucket_name=bucket_name, object_name=object_name, file_path=file_path)
                #     except:
                #         print('Upload error')

    def pull_bucket(self, bucket_name, destination_dir):
        objects = self.client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            object_name = obj.object_name
            destination_path = os.path.join(destination_dir, object_name)
            self.client.fget_object(bucket_name, object_name, destination_path)


if __name__ == '__main__':
    minio_obj = Bucket()
    minio_obj.bucket_list_files("b2809084cfd2f2ecd28db8172d3c5cd6",None)
