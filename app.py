import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("💬 WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # ── TOP STATISTICS ────────────────────────────────────────────────────
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("📊 Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", num_messages)
        with col2:
            st.metric("Total Words", words)
        with col3:
            st.metric("Media Shared", num_media_messages)
        with col4:
            st.metric("Links Shared", num_links)

        st.markdown("---")

        # ── MONTHLY TIMELINE ─────────────────────────────────────────────────
        st.title("📅 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        if timeline.empty:
            st.info("No data available.")
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(timeline['time'], timeline['message'], color='green', marker='o', linewidth=2)
            ax.fill_between(range(len(timeline)), timeline['message'], alpha=0.1, color='green')
            plt.xticks(range(len(timeline)), timeline['time'], rotation='vertical')
            ax.set_ylabel("Messages")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)

        # ── DAILY TIMELINE ───────────────────────────────────────────────────
        st.title("📆 Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        if daily_timeline.empty:
            st.info("No data available.")
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black', linewidth=1.5)
            ax.fill_between(daily_timeline['only_date'], daily_timeline['message'], alpha=0.1, color='black')
            plt.xticks(rotation='vertical')
            ax.set_ylabel("Messages")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)

        st.markdown("---")

        # ── ACTIVITY MAP ─────────────────────────────────────────────────────
        st.title("🗺️ Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            bars = ax.bar(busy_day.index, busy_day.values, color='purple')
            ax.bar_label(bars)
            plt.xticks(rotation='vertical')
            ax.set_ylabel("Messages")
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            bars = ax.bar(busy_month.index, busy_month.values, color='orange')
            ax.bar_label(bars)
            plt.xticks(rotation='vertical')
            ax.set_ylabel("Messages")
            st.pyplot(fig)

        # ── WEEKLY ACTIVITY HEATMAP ───────────────────────────────────────────
        st.title("🔥 Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        if user_heatmap.empty or user_heatmap.size == 0:
            st.info("Not enough data to display the weekly activity heatmap.")
        else:
            fig, ax = plt.subplots(figsize=(14, 5))
            sns.heatmap(user_heatmap, ax=ax, cmap='YlOrRd', linewidths=0.5, annot=True, fmt='.0f')
            st.pyplot(fig)

        st.markdown("---")

        # ── NEW: HOURLY DISTRIBUTION ──────────────────────────────────────────
        st.title("⏰ Hourly Message Distribution")
        hourly = helper.hourly_message_distribution(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(hourly['hour'], hourly['message'], color='steelblue')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Messages")
        ax.set_xticks(range(24))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        # ── NEW: TIME OF DAY BREAKDOWN ────────────────────────────────────────
        st.title("🌅 Messages by Time of Day")
        period_data = helper.messages_by_period_of_day(selected_user, df)
        fig, ax = plt.subplots()
        colors = ['#FFD700', '#FF8C00', '#FF4500', '#191970']
        ax.pie(period_data.values, labels=period_data.index, autopct='%1.1f%%',
               colors=colors, startangle=140)
        ax.set_title("Morning / Afternoon / Evening / Night")
        st.pyplot(fig)

        st.markdown("---")

        # ── NEW: SENTIMENT OVER TIME ──────────────────────────────────────────
        st.title("😊 Sentiment Trend Over Time")
        sentiment = helper.sentiment_over_time(selected_user, df)
        if sentiment.empty:
            st.info("Not enough data.")
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(sentiment['time'], sentiment['positive'], color='green',
                    marker='o', label='Positive', linewidth=2)
            ax.plot(sentiment['time'], sentiment['negative'], color='red',
                    marker='o', label='Negative', linewidth=2)
            plt.xticks(range(len(sentiment)), sentiment['time'], rotation='vertical')
            ax.set_ylabel("Word Count")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)

        st.markdown("---")

        # ── NEW: AVERAGE MESSAGE LENGTH ───────────────────────────────────────
        st.title("📝 Average Message Length by User")
        avg_len = helper.avg_message_length(selected_user, df)
        if avg_len.empty:
            st.info("Not enough data.")
        else:
            fig, ax = plt.subplots()
            bars = ax.barh(avg_len['user'], avg_len['msg_len'], color='teal')
            ax.bar_label(bars, fmt='%.1f')
            ax.set_xlabel("Avg Words per Message")
            st.pyplot(fig)

        # ── NEW: RESPONSE TIME ────────────────────────────────────────────────
        st.title("⚡ Average Response Time (minutes)")
        resp = helper.response_time_analysis(df)
        if resp.empty:
            st.info("Not enough data.")
        else:
            fig, ax = plt.subplots()
            bars = ax.barh(resp['user'], resp['gap_min'], color='coral')
            ax.bar_label(bars, fmt='%.1f')
            ax.set_xlabel("Avg Response Time (min)")
            st.pyplot(fig)

        st.markdown("---")

        # ── BUSIEST USERS (Group only) ────────────────────────────────────────
        if selected_user == 'Overall':
            st.title('👥 Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                bars = ax.bar(x.index, x.values, color='red')
                ax.bar_label(bars)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # ── WORDCLOUD ─────────────────────────────────────────────────────────
        st.title("☁️ Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        if df_wc:
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Not enough text to generate a word cloud.")

        # ── MOST COMMON WORDS ─────────────────────────────────────────────────
        most_common_df = helper.most_common_words(selected_user, df)
        if not most_common_df.empty:
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1], color='slateblue')
            ax.set_xlabel("Frequency")
            st.title('🔤 Most Common Words')
            st.pyplot(fig)

        # ── EMOJI ANALYSIS ────────────────────────────────────────────────────
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("😂 Emoji Analysis")
        if not emoji_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df.rename(columns={0: 'Emoji', 1: 'Count'}))
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(8), labels=emoji_df[0].head(8), autopct="%0.1f%%")
                st.pyplot(fig)
        else:
            st.info("No emojis found for the selected user.")