import re
import pandas as pd

def preprocess(data):
    # Format: M/D/YY, HH:MM AM/PM -  (e.g. 1/4/26, 11:24 AM - )
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][Mm]\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    min_len = min(len(messages), len(dates))
    messages = messages[:min_len]
    dates = dates[:min_len]

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Your format: 1/4/26, 11:24 AM - 
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages_list = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages_list.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages_list.append(entry[0])

    df['user'] = users
    df['message'] = messages_list
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + "-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df