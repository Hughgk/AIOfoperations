from handwriting_calculator.mongua import Mongua



class User(Mongua):
    #MongoDB
    __fields__ = Mongua.__fields__ + [
        ('username', str, ''),
        ('password', str, ''),
        ('user_image', str, 'photo.jpg'), #用户头像
    ]

    """
        User 是一个保存用户数据的 model
        现在只有两个属性 username 和 password
        """

    def __init__(self):
        #设置用户默认头像
        self.user_image = 'photo.jpg'

    #网络安全性
    def salted_password(self, password, salt='$!@><?>HUI&DWQa`'):
        import hashlib
        def sha256(ascii_str):
            return hashlib.sha256(ascii_str.encode('ascii')).hexdigest()

        hash1 = sha256(password)
        hash2 = sha256(hash1 + salt)
        return hash2


    @classmethod
    def register(cls, form):
        name = form.get('username', '')
        pwd = form.get('password', '')
        if len(name) > 2 and User.find_by(username=name) is None:
            u = User.new(form)
            u.password = u.salted_password(pwd)
            u.save()
            return u
        else:
            return None

    @classmethod
    def validate_login(cls, form):
        u = User()
        u.username = form.get("username", '')
        u.password = form.get("password", "")
        user = User.find_by(username=u.username)
        if user is not None and user.password == u.salted_password(u.password):
            return user
        else:
            return None
