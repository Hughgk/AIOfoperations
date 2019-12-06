import time
from pymongo import MongoClient
mongua = MongoClient()




def next_id(name):
    query = {
        'name': name,
    }
    update = {
        '$inc': {
            'seq': 1
        }
    }
    kwargs = {
        'query': query,
        'update': update,
        'upsert': True,
        'new': True,
    }
    # 存储数据的 id
    doc = mongua.data['data_id']
    # find_and_modify 是一个原子操作函数
    new_id = doc.find_and_modify(**kwargs).get('seq')
    return new_id


class Mongua(object):
    __fields__ = [
        '_id',
        # (字段名, 类型, 值)
        ('id', int, -1),           #数据的序号
        ('type', str, ''),         #该数据的类型 ： user ， topic ， board
        ('deleted', bool, False),  #将数据默认标记为未删除状态 ， 如果已经删除的数据则该字段会变为True
        ('created_time', int, 0),  #创建时间
        ('updated_time', int, 0),  #最近更新时间
    ]





    @classmethod
    def new(cls, form=None, **kwargs):
        """
        new 是给外部使用的函数
        """
        name = cls.__name__
        # 创建一个空对象
        m = cls()
        # 把定义的数据写入空对象, 未定义的数据输出错误
        fields = cls.__fields__.copy()
        # 去掉 _id 这个特殊的字段
        fields.remove('_id')
        if form is None:
            form = {}

        for f in fields:
            k, t, v = f
            if k in form:
                setattr(m, k, t(form[k]))
            else:
                # 设置默认值
                setattr(m, k, v)
        # 处理额外的参数 kwargs
        for k, v in kwargs.items():
            if hasattr(m, k):
                setattr(m, k, v)
            else:
                raise KeyError
        # 写入默认数据
        m.id = next_id(name)
        # print('debug new id ', m.id)
        ts = int(time.time())
        m.created_time = ts
        m.updated_time = ts
        # m.deleted = False
        m.type = name.lower()
        # 特殊 model 的自定义设置
        # m._setup(form)
        m.save()
        return m

    @classmethod
    def _new_with_bson(cls, bson):
        """
        这是给内部 all 这种函数使用的函数
        从 mongo 数据中恢复一个 model
        """
        m = cls()
        fields = cls.__fields__.copy()
        # 去掉 _id 这个特殊的字段
        fields.remove('_id')
        for f in fields:
            k, t, v = f
            if k in bson:
                setattr(m, k, bson[k])
            else:
                # 设置默认值
                setattr(m, k, v)
        setattr(m, '_id', bson['_id'])
        # 这一句必不可少，否则 bson 生成一个新的_id
        # FIXME, 因为现在的数据库里面未必有 type
        # 所以在这里强行加上
        # 以后洗掉db的数据后应该删掉这一句
        m.type = cls.__name__.lower()
        return m

    @classmethod
    def all(cls):
        return cls._find()

    # TODO, 还应该有一个函数 find(name, **kwargs)
    @classmethod
    def _find(cls, **kwargs):
        """
        mongo 数据查询
        """
        name = cls.__name__
        # TODO 过滤掉被删除的元素
        kwargs['deleted'] = False
        flag_sort = '__sort'
        sort = kwargs.pop(flag_sort, None)
        ds = mongua.data[name].find(kwargs)
        if sort is not None:
            ds = ds.sort(sort)
        l = [cls._new_with_bson(d) for d in ds]
        return l


    @classmethod
    def find_by(cls, **kwargs):
        return cls.find_one(**kwargs)

    @classmethod
    def find_all(cls, **kwargs):
        return cls._find(**kwargs)

    @classmethod
    def find(cls, id):
        return cls.find_one(id=id)

    @classmethod
    def get(cls, id):
        return cls.find_one(id=id)

    @classmethod
    def find_one(cls, **kwargs):
        """
        """
        # TODO 过滤掉被删除的元素
        kwargs['deleted'] = False
        l = cls._find(**kwargs)
        # print('find one debug', kwargs, l)
        if len(l) > 0:
            return l[0]
        else:
            return None


    def save(self):
        name = self.__class__.__name__
        mongua.data[name].save(self.__dict__)

    def delete(self):
        name = self.__class__.__name__
        print("mongua-delete : " + str(self))
        print("delete-name :" + name)
        query = {
             "id": self.id,
        }

        values = {
            "$set": {"deleted": True},
        }
        res = mongua.data[name].update_one(query, values)
        print(res, res.modified_count)

