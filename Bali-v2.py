corpus = pd.read_pickle('corpus.pkl')
corpus_embeddings = pd.read_pickle('corpus_embeddings.pkl')
aggregate_reviews = pd.read_pickle('aggregate_reviews.pkl')
aggregate_summary = pd.read_pickle('aggregate_summary.pkl')

embedder = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

userinput = st.text_input('What kind of hotel are you looking for?')
if not userinput:
    st.write("Please enter a query to get results")
else:
    query = [str(userinput)]
    # query_embeddings = embedder.encode(queries,show_progress_bar=True)
    top_k = min(2, len(corpus))

    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.write("\n\n======================\n\n")
    st.write("Query:", query)
    st.write("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        st.write("(Score: {:.4f})".format(score))
        row_dict = aggregate_reviews.loc[aggregate_reviews['review_body'] == corpus[idx]]['hotelName'].values[0]
        summary = aggregate_summary.loc[aggregate_summary['review_body'] == corpus[idx]]['summary']
        st.write("paper_id:  " , row_dict['hotel_name'] , "\n")
        #wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='navy', colormap='rainbow', collocations=False, stopwords = STOPWORDS, mask=mask).generate(corpus[idx])
        wordcloud = WordCloud(collocations=False,stopwords=stopwords,background_color='black',max_words=35).generate(corpus[idx])
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(fig)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    # for score, idx in zip(top_results[0], top_results[1]):
    #     row_dict = aggregate_reviews.loc[aggregate_reviews['review_body'] == corpus[idx]]['hotelName'].values[0]
    #     summary = aggregate_summary.loc[aggregate_summary['review_body'] == corpus[idx]]['summary']
    #     st.write(HTML_WRAPPER.format(
    #         "<b>Hotel Name:  </b>" + re.sub(r'[0-9]+', '', row_dict) + "(Score: {:.4f})".format(
    #             score) + "<br/><br/><b>Hotel Summary:  </b>" + summary.values[0]), unsafe_allow_html=True)
    #     self.plot_wordCloud(corpus[idx])