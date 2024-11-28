from elasticsearch import Elasticsearch

class ESClient:
    def __init__(self, host:str="x.x.x.x:9200", user:str="elastic", password:str="elastic", timeout:int=50):
        """
        Initialize the Elasticsearch client.
        :param host: Elasticsearch host (e.g., "http://127.0.0.1:9200")
        :param user: Username for HTTP authentication (optional)
        :param password: Password for HTTP authentication (optional)
        :param timeout: Request timeout in seconds (default is 50)
        """
        self.client = Elasticsearch(
            hosts=[host],
            timeout=timeout,
            http_auth=(user, password) if user and password else None
        )

    def search(self, index, query):
        """
        Perform a search query on the specified index.

        :param index: Elasticsearch index to search in
        :param query: Query body in Elasticsearch DSL format
        :return: Search results
        """
        response = self.client.search(index=index, body=query)
        return response['hits']['hits']

    def match_phrase_query(self, index, fields, values):
        """
        Perform a match_phrase query on the specified index.
        index = "document_shard_index_lm"
        fields = ["content", "content", "content"]
        values = ["9月10日", "澳", "大狗"]
        :param index: Elasticsearch index to search in
        :param fields: List of fields to search in
        :param values: List of values to search for (must match the length of fields)
        :return: Search results
        """
        if len(fields) != len(values):
            raise ValueError("The number of fields must match the number of values")

        query = {
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        {"match_phrase": {field: value}} for field, value in zip(fields, values)
                    ]
                }
            }
        }

        return self.search(index, query)

# Example usage
if __name__ == '__main__':
    # Configuration
    host = "http://127.0.0.1:9200"
    user = "elastic"
    password = "elastic"

    # Initialize the ESClient
    es_client = ESClient(host, user=user, password=password)

    # Define the query parameters
    index = "document_shard_index_lm"
    fields = ["content", "content", "content"]
    values = ["9月10日", "澳", "大狗"]

    # Perform the search
    results = es_client.match_phrase_query(index, fields, values)
    print(results)