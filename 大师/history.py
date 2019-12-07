import time
import json
from handwriting_calculator.user import User
from handwriting_calculator.mongua import Mongua
import logging
import os
import time




class History(Mongua):
    __fields__ = Mongua.__fields__ + [
        ('content', str, ''),   #四则运算
        ('user_id', int, -1),   #所属用户ID
    ]


    def user(self):
        #发表话题用户
        u = User.find(id=self.user_id)
        return u
