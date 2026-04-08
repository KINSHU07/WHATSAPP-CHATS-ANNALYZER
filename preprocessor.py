import re
import pandas as pd

def preprocess(data):
    # Supports both 12-hour (AM/PM) and 24-hour WhatsApp timestamp formats
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[APap][Mm])?\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Trim lists to equal length (safety guard)
    min_len = min(len(messages), len(dates))
    messages = messages[:min_len]
    dates = dates[:min_len]

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Try multiple date formats to handle different WhatsApp exports
    parsed = None
    for fmt in (
        '%d/%m/%Y, %H:%M - ',
        '%d/%m/%y, %H:%M - ',
        '%m/%d/%Y, %H:%M - ',
        '%m/%d/%y, %H:%M - ',
        '%d/%m/%Y, %I:%M %p - ',
        '%d/%m/%y, %I:%M %p - ',
    ):
        try:
            parsed = pd.to_datetime(df['message_date'], format=fmt)
            break
        except (ValueError, TypeError):
            continue

    # Fallback: format='mixed' works in pandas 2.0+ (replaces removed infer_datetime_format)
    if parsed is None:
        parsed = pd.to_datetime(df['message_date'], format='mixed', errors='coerce')

    df['message_date'] = parsed
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages_list = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:  # user name present
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