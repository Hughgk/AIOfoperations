# !/usr/bin/env python

# -*- coding: utf-8 -*-
import pyodbc


class MSSQL:
    """
    封装pyodbc
    """

    def __init__(self, host, user, pwd, db="Data", charset="utf8"):

        self._host = host

        self._user = user

        self._pwd = pwd

        self._db = db

        self._charset = charset

    def __get_connect(self):

        """
        得到连接信息
        返回: conn.cursor()
        """

        if not self._db:
            raise (NameError, "没有设置数据库信息")

        conn_info = "DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s" % (
            self._host, self._db,  self._user, self._pwd)

        self.conn = pyodbc.connect(conn_info, charset=self._charset)

        cur = self.conn.cursor()

        if not cur:

            raise (NameError, "连接数据库失败")

        else:

            return cur


    def __exec_query(self, sql):
        """
        执行查询语句
        返回一个包含tuple的list，list的元素是记录行，tuple的元素是每行记录的字段
        """

        cur = self.__get_connect()

        cur.execute(sql)

        resList = cur.fetchall()

        # 查询完毕后必须关闭连接

        self.conn.close()

        return resList


    def exec_query_tuple(self, sql):
        """
        结果集以元组返回
        """

        return self.__exec_query(sql)


    def exec_query_dict(self, sql):
        result = []

        for row in self.__exec_query(sql):
            result.append(dict([(desc[0], row[index]) for index, desc in enumerate(row.cursor_description)]))

        return result


    def exec_nonquery(self, sql):
        """
        执行非查询语句
        """

        cur = self.__get_connect()

        cur.execute(sql)

        self.conn.commit()

        self.conn.close()


if __name__ == "__main__":
    conn = MSSQL("localhost", "sa", "123456", "Data", "GBK")
    print(conn.exec_query_dict("select * from user_table"))


