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


# ── NEW FEATURES ─────────────────────────────────────────────────────────────

def hourly_message_distribution(selected_user, df):
    """Messages count by hour of day (0-23)."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('hour').count()['message'].reset_index()


def sentiment_over_time(selected_user, df):
    """
    Very lightweight sentiment proxy: counts positive words vs negative words
    per month using simple keyword lists (no extra library needed).
    Returns a dataframe with columns: time, positive, negative.
    """
    positive_words = {'good','great','happy','love','nice','awesome','fantastic',
                      'wonderful','best','excellent','amazing','beautiful','fun',
                      'enjoy','glad','thanks','thank','lol','haha','😊','❤','🥰',
                      '😍','🎉','👍','😂','🙏'}
    negative_words = {'bad','sad','hate','worst','terrible','awful','horrible',
                      'boring','angry','upset','annoyed','stressed','cry','miss',
                      'sorry','😢','😭','😠','😡','💔','😞','🙁','😔'}

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification'].copy()
    temp['positive'] = temp['message'].apply(
        lambda m: sum(1 for w in m.lower().split() if w in positive_words))
    temp['negative'] = temp['message'].apply(
        lambda m: sum(1 for w in m.lower().split() if w in negative_words))

    result = temp.groupby(['year', 'month_num', 'month'])[['positive','negative']].sum().reset_index()
    result['time'] = result['month'] + '-' + result['year'].astype(str)
    return result


def avg_message_length(selected_user, df):
    """Average message length (words) per user."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n'].copy()
    temp['msg_len'] = temp['message'].apply(lambda m: len(m.split()))
    return temp.groupby('user')['msg_len'].mean().sort_values(ascending=False).reset_index()


def response_time_analysis(df):
    """
    Average response time (minutes) between consecutive messages per user.
    Only considers gaps < 60 min to ignore overnight silences.
    """
    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp.sort_values('date').reset_index(drop=True)
    temp['prev_time'] = temp['date'].shift(1)
    temp['prev_user'] = temp['user'].shift(1)
    temp['gap_min'] = (temp['date'] - temp['prev_time']).dt.total_seconds() / 60

    # Only count when a DIFFERENT user replies within 60 minutes
    replies = temp[(temp['user'] != temp['prev_user']) & (temp['gap_min'] < 60)]
    return replies.groupby('user')['gap_min'].mean().sort_values().reset_index()


def messages_by_period_of_day(selected_user, df):
    """Categorise messages into Morning / Afternoon / Evening / Night."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    def get_period(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    temp = df[df['user'] != 'group_notification'].copy()
    temp['time_of_day'] = temp['hour'].apply(get_period)
    order = ['Morning', 'Afternoon', 'Evening', 'Night']
    result = temp['time_of_day'].value_counts().reindex(order).fillna(0)
    return result