import requests
import json
from src.user_operations import UserOperations

class Gemini():
    def __init__(self,config):
        self.config = config if config else {}
        self.user_operations = UserOperations()
        self.gemini_api_key = config['gemini_api_key']  # Replace with your actual API key

    def query_gemini(self, query):
        import requests
        import json
        
        # Set up your API key and the URL for the request
        api_key = self.gemini_api_key  # Replace with your actual API key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        query = 'test'
        # Create the payload (the data you want to send)
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{query}"
                        }
                    ]
                }
            ]
        }
        
        # Set the headers to specify content type
        headers = {
            "Content-Type": "application/json"
        }
        
        # Send the POST request to the API
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Check the response status and print the result
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
    
    def parse_first_text(self,response_json):
        print(response_json['candidates'][0]['content']['parts'][0]['text'])


    def ask_for_reccomendation(self, read_book_ids, vector_recomnedation_book_id, param_book_df):
        """
            Ask Gemini for a reccomnedation for a book given the followin parameters:
                    
            :param read_book_ids:  that have been read by the user
            :param vector_recomnedation_book_id: ID of the book that had the closest cosine similiarity to the user's vector
            :param_book_df: DF of all possible books. ** May need to remove several columns from original to reduce token count **
            :return: Book reccomended by Gemini
        """
        query = f"""
            You are a book recommendation assistant. Your job is to select the most entertaining book for for the user. 
            You will be given a dataframe of books that you can choose from. This dataframe may have some books that you know, so use your knowledge about the book as well as it's given metadata. Do not discriminate against books you don't know.
            
            You will also given the ids to the books that the user has already read, these id's will coorelate to the df you are given. You may want to influence your decision based on what the user has already read and what kind of pattern you see.
            
            You will also be given the id of the book that a vectorization algorithm chose. You may want to influence your decision using this information as well. It's a rudimentary algorithm so this may not necesarily be the correct choice either.
    
    
            Please give your best recommendation for the next book a user should read. Here are 3 parameters you can use, which have been described in the previous text: 
            
            :param read_book_ids:  {read_book_ids}
            :param vector_recomnedation_book_id: {vector_recomnedation_book_id}
            :param_book_df: {param_book_df}
        """
    
        self.query_gemini(query)
        






    