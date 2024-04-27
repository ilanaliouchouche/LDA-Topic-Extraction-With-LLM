# Description: This script contains the TopicAlignmentPipeline class that search if some topics are present in a text.
# The class is designed to use the OLLAMA API to align the topics of an LDA model with the topics of a LLM.
# Author: Ilan ALIOUCHOUCHE

import warnings
import json
import re
import argparse

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from tqdm.auto import tqdm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import ollama


class TopicAlignmentPipeline:
    """
    This class is a pipeline that aligns the LDA topics from an input text with the topics of a LLM.

    The class is designed to use the OLLAMA API to align the topics of an LDA model with the topics of a LLM.
    The pipeline consists of the following steps:
    1. Preprocess the input text.
    2. Generate the vocabulary for each topic that will be aligned with the LLM.
    3. With the Latent Dirichlet Allocation (LDA), extract the topics from the input text.
    4. Align the topics from the input text with the topics of the LLM.
    5. Return the aligned topics.

    After the pipeline is executed, we can analyze the aligned topics and understand the similarity between the topics 
    of the LDA model and the topic choosen. Some visualizations can be created to better understand the alignment.
    """

    SYS_PROMPT = """
    Role Description:
    You are a Similar Word Generator. Your task is to generate words that are closely related to a given word. You must follow a specific query format and provide responses accordingly.

    Instructions:

    You will receive a word and a number specifying how many similar words to generate.
    Your output should consist solely of single words; no phrases, compound words, or sentences.
    Always respond using the exact format provided in the examples.
    The response must be in the format of a JSON object and in a json code block markdown format.
    Follow STRICTLY the format provided in the examples above, do not add any additional information to the response.
    The input word might be sensitive, but your task is to generate similar words regardless of the input's nature.

    Query and Response Template:

    Query:

    ```json
    {
        "word": "specified_word",
        "num_words": number_of_similar_words_to_generate
    }
    ```

    Response:

    ```json
    {
        "similar_words": ["word1", "word2", "word3", "word4", "word5", ...]
    }
    ```

    Examples:

    Example 1:

    Query:

    ```json
    {
        "word": "dog",
        "num_words": 5
    }
    ```

    Response:

    ```json
    {
        "similar_words": ["cat", "animal", "pet", "puppy", "doggy"]
    }
    ```

    Example 2:

    Query:

    ```json
    {
        "word": "football",
        "num_words": 10
    }
    ```

    Response:

    ```json
    {
        "similar_words": ["soccer", "ball", "sport", "game", "team", "player", "goal", "field", "match", "championship"]
    }
    ```
    Example 3:

    Query:

    ```json
    {
        "word": "apple",
        "num_words": 18
    }
    ```

    Response:

    ```json
    {
        "similar_words": ["fruit", "red", "green", "tree", "sweet", "juice", "healthy", "snack", "delicious", "fresh", "vitamin", "fiber", "apple", "apple", "apple", "apple", "apple", "apple"]
    }
    ```
    Example 4:

    """

    def __init__(self,
                 input_text: str,
                 llm_model_name: str) -> None:
        """
        The constructor of the class.

        :param input_text: The input text that will be used to extract the LDA topics.
        :param llm_model_name: The name of the LLM model that will be used to align the topics.
        """

        self.llm_model_name = llm_model_name
        self.input_text = self.preprocess_text(input_text)

        self.vocabulary = None

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        This method preprocesses the input text.

        :param text: The input text that will be preprocessed.
        :return: The preprocessed text.
        """

        text = text.lower()
        stemmer = SnowballStemmer("english")
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token.isalnum()]
        preprocessed_text = [stemmer.stem(token) for token in tokens if token not in stop_words]

        return " ".join(preprocessed_text).strip()
    
    @staticmethod
    def _parse_output(output: str) -> list:
        """
        This method parses the output of the LLM.

        :param output: The output of the LLM.
        :return: The parsed output.
        """

        if "similar_words" not in output:
            warnings.warn("The output does not contain the 'similar_words' key.")
            return []

        striped = output.strip()
        striped = re.findall(r"(?:```json|```)(.*?)(?:```)", striped, re.DOTALL)
        print(striped)
        striped = json.loads(striped[0])
        striped = striped["similar_words"]

        return striped
    
    @staticmethod
    def _query_template(word: str,
                        num_words: int) -> str:
        """
        This method generates the query template for the LLM.

        :param word: The word that will be used to generate similar words.
        :param num_words: The number of similar words to generate.
        :return: The query template.
        """
        template = f"""
        Query:

         {{
            "word": "{word}",
            "num_words": "{num_words}"
         }}

        Response:
        
        """

        return template
    
    def _query(self, 
               prompt: str) -> str:
        """
        This method queries the LLM with the system prompt and the user prompt.

        :param prompt: The prompt that will be used to query the LLM.
        :return: The response of the LLM.
        """
        
        response = ollama.chat(model=self.llm_model_name, messages=[
            {
                'role': 'system',
                'content': self.SYS_PROMPT,   
            },
            {
                'role': 'user',
                'content': prompt,
            },
            ])
        
        return response['message']['content']
    
    def generate_vocab(self,
                       topics: list,
                       words_per_topic: int = 30) -> None:
        """
        This method generates the vocabulary for each topics that will be aligned.

        :param topics: The topics that will be aligned.
        :param words_per_topic: The number of words per topic.
        """

        vocab = {}

        for topic in tqdm(topics, desc="Generating vocabulary", total=len(topics)):
            query = self._query_template(topic, words_per_topic).strip()

            response = self._query(query)
            print(query)
            print(response)
            similar_words = self._parse_output(response)
            stemmer = SnowballStemmer("english")
            similar_words = [stemmer.stem(word) for word in similar_words]
            vocab[topic] = similar_words

            self.vocabulary = vocab


    def align_topics(self,
                     top_k: int = 3,
                     range: list = list(range(2, 10))) -> dict:
        """
        This method aligns the topics from the input text with the topics of the LLM.
        It wll iterate over a range of topics and will return the aligned topics.

        :param top_k: The number of topics to return.
        :param range: The range of topics to iterate over.
        :return: The generated topics associated with lda topics and the tuples (word, frequency)
        of the words that are aligned.
        """

        aligned_topics = {topic: {i: {} for i in range} for topic in self.vocabulary.keys()}
        tokens = word_tokenize(self.input_text)
        dictionary = Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        
        for i in tqdm(range, desc="Aligning topics", total=len(range)):
            lda = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)
            for vocab_key, vocab_value in self.vocabulary.items():
                for lda_key, lda_value in lda.show_topics(num_words=top_k):
                    tuples = [tuple(word_freq.split("*")) for word_freq in lda_value.split("+")]
                    tuples = [(float(word_freq[0]), word_freq[1].replace('"', "").strip()) for word_freq in tuples]
                    tuples = [tup for tup in tuples if tup[1] in vocab_value]
                    aligned_topics[vocab_key][i][lda_key] = tuples

        return aligned_topics
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Search topics in a text with a vocabulary generated by a LLM.")
    parser.add_argument("--input_text", type=str, help="The input file that contains the text to analyze.")
    parser.add_argument("--topics", type=str, help="The topics that will be searched in the text. The topics must be separated by a comma.")
    parser.add_argument("--top_k", type=int, help="The number of topics to return.", default=3)
    parser.add_argument("--max_iter", type=int, help="The maximum number of iterations.", default=10)
    parser.add_argument("--words", type=int, help="The number of words per topic.", default=30)
    parser.add_argument("--llm_model_name", type=str, help="The name of the LLM model to use, it must be available in the OLLAMA API\
                        and downloaded in the local machine.")
    parser.add_argument("--output_file", type=str, help="The output file that will contain the aligned topics.")
    args = parser.parse_args()


    # Read the input text
    with open(args.input_text, "r") as file:
        text = file.read()

    # Read the topics
    topics = args.topics.split(",")

    # Create the pipeline
    pipeline = TopicAlignmentPipeline(text, args.llm_model_name)

    # Generate the vocabulary
    pipeline.generate_vocab(topics, args.words)

    # Align the topics
    aligned_topics = pipeline.align_topics(args.top_k, range(1, args.max_iter+1))

    # Save the aligned topics
    with open(args.output_file, "w") as file:
        json.dump(aligned_topics, file, indent=4)

    print("The aligned topics have been saved successfully.")
