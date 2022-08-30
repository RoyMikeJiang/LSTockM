import pymysql

async def sql_fetch_json(cursor: pymysql.cursors.Cursor):
    """
    Convert the pymysql SELECT result to json format
    :param cursor:
    :return:
    """
    keys = []
    for column in cursor.description:
        keys.append(column[0])
    key_number = len(keys)

    json_data = []
    for row in cursor.fetchall():
        item = dict()
        for q in range(key_number):
            item[keys[q]] = row[q]
        json_data.append(item)

    return json_data

async def sql_fetch_list(cursor: pymysql.cursors.Cursor):
    """
    Convert the pymysql SELECT result to list format
    :param cursor:
    :return:
    """
    keys = []
    for column in cursor.description:
        keys.append(column[0])
    key_number = len(keys)

    list_data = []

    if key_number == 1:
        for row in cursor.fetchall():
            list_data.append(row[0])
        return list_data

    for row in cursor.fetchall():
        item = []
        for q in range(key_number):
            item.append(row[q])
        list_data.append(item)
    return list_data