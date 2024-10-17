#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:quincy qiang
# email:yanqiangmiffy@gmail.com
# datetime:2021/5/25 11:05
# description:"时间相关函数"
import time
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta


def get_dt2ts(time_str):
    """
    将日期转为时间戳 单位ms
    :param time_str: 2020-12-22 00:00:00 or 2020-12-22
    :return:1608566400000
    """
    if len(time_str) == 19:
        datetime_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    elif len(time_str) == 10:
        datetime_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d")
    else:
        datetime_obj = None
        print("日期格式不正确")
    obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return obj_stamp


def convert_ts2dt(timestamp=1619193600000, is_ms=False):
    """
    将时间戳转为日期
    :param timestamp:
    :return:
    """
    # timestamp = 1606320000000
    time_local = time.localtime(timestamp / 1000)
    # 转换成新的时间格式(2016-05-05 20:28:54)
    if is_ms:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    else:
        dt = time.strftime("%Y-%m-%d", time_local)
    return dt


def get_next_day(begin_date):
    """
    获取输入日期的下一天日期
    :param begin_date:2021-05-24
    :return:2021-05-25
    """
    if len(begin_date) == 19:
        dt = datetime.datetime.strptime(begin_date, "%Y-%m-%d %H:%M:%S")
    elif len(begin_date) == 10:
        dt = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    else:
        dt = None
        print("日期格式不正确")
    dt = dt + datetime.timedelta(1)
    end_date = dt.strftime("%Y-%m-%d")
    return end_date


def get_standardtime_by_offset(
        date='2021-04-24',
        type=1,
        year=0,
        month=0,
        day=0,
        hour=0,
        minute=0,
        second=0,
):
    '''
    根据现在时间和设定偏移量获取标准时间
    :param type:偏移类型，1为加法，其他为减法
    :param year:年
    :param month:月
    :param day:日
    :param hour:小时
    :param minute:分钟
    :param second:秒
    :return:如1970-01-01 00:00:00
    '''
    if len(date) == 19:
        dt = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    elif len(date) == 10:
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    else:
        dt = None
        print("日期格式不正确")
    if type == 1:
        return (dt + relativedelta(
            years=year,
            months=month,
            days=day,
            hours=hour,
            minutes=minute,
            seconds=second
        )).strftime("%Y-%m-%d %H:%M:%S")
    else:
        return (dt - relativedelta(
            years=year,
            months=month,
            days=day,
            hours=hour,
            minutes=minute,
            seconds=second
        )).strftime(
            "%Y-%m-%d %H:%M:%S")


def get_dates_range(start_date='2020-12-10', end_data='2020-12-20'):
    """
    获取开始日期和结束日期所有的日期列表，包括边界日期
    :param start_date:2020-12-10
    :param end_data:2020-12-20
    :return:['2020-12-10',....'2020-12-19','2020-12-20]
    """
    return pd.date_range(start=start_date, end=end_data).astype(str).values.tolist()


def get_today():
    """
    返回今天日期 2021-05-24
    :return:2021-05-24
    """
    today = time.strftime('%Y-%m-%d', time.localtime())
    return today


def get_now_time():
    """
    获取当前时间
    :return:
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def get_window_days(end_date='2020-12-21', window_size=10):
    """
    获取距离日期end_date的前window_size的历史日期
    :param end_date:2020-12-21
    :param window_size:10
    :return:['2020-12-11', '2020-12-12', '2020-12-13', '2020-12-14', '2020-12-15',
            '2020-12-16', '2020-12-17', '2020-12-18', '2020-12-19', '2020-12-20']
    """
    return pd.date_range(end=end_date, periods=window_size + 1).astype(str).values.tolist()[:-1]


def get_today_lastdays(period=10):
    """

    :param period:
    :return:
    """
    today = time.strftime('%Y-%m-%d', time.localtime())
    res = get_window_days(today, window_size=period)
    return res


if __name__ == '__main__':
    print("日期转时间戳", get_dt2ts('2020-12-22 00:00:00'))
    print("时间戳转日期", convert_ts2dt(1619193600000))
    print("获取输入日期的下一天日期", get_next_day('2021-05-24'))
    print("获取指定偏移量的未来日期", get_standardtime_by_offset(date='2021-05-24', day=1))
    print("获取时间列表", get_dates_range('2020-12-10', '2020-12-20'))
    print("获取今天日期", get_today())
    print("获取当前时间", get_now_time())
    print("获取指定日期的历史窗口日期列表", get_window_days())
    print("获取今天的历史窗口日期列表", get_today_lastdays())
