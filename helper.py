from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    df.columns = ['name', 'percent']
    return x, df


def create_wordcloud(selected_user, df):
    try:
        f = open('stop_hinglish.txt', 'r')
        stop_words = f.read()
        f.close()
    except FileNotFoundError:
        stop_words = ""

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n'].copy()

    if temp.empty:
        return None

    def remove_stop_words(message):
        return " ".join(
            word for word in message.lower().split()
            if word not in stop_words
        )

    temp['message'] = temp['message'].apply(remove_stop_words)
    combined = temp['message'].str.cat(sep=" ").strip()

    if not combined:
        return None

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(combined)


def most_common_words(selected_user, df):
    try:
        f = open('stop_hinglish.txt', 'r')
        stop_words = f.read()
        f.close()
    except FileNotFoundError:
        stop_words = ""

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = [
        word
        for message in temp['message']
        for word in message.lower().split()
        if word not in stop_words
    ]

    return pd.DataFrame(Counter(words).most_common(20))


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        # FIX: emoji.UNICODE_EMOJI is deprecated; use emoji.is_emoji() instead
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(
        index='day_name', columns='period',
        values='message', aggfunc='count'
    ).fillna(0)