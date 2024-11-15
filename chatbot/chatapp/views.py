from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt  
from groq import Groq
from chatapp.utils.pydantic_files import LLMResponse
from chatapp.utils.rag import store_embeddings, semantic_search
import json
from django.http import JsonResponse
from dotenv import load_dotenv
load_dotenv()

def store_embeddings_api(request):
    store_embeddings()


@csrf_exempt
def llm_call(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))  
            query = data.get("query")  
            
            if not query:
                return JsonResponse({"error": "Query parameter is required."}, status=400)


            rag_results = semantic_search(query)
            
            context = "\n".join([result.content for result in rag_results])
            
            system_prompt = """You are a helpful AI assistant. Using the provided context, 
            answer the user's question accurately and concisely. If the context doesn't 
            contain relevant information, acknowledge that and provide a general response.
            Make sure that you avoid political topics, your response should not contain any details about
            any political party, if recieved context has any political data in it, ignore it.
            Elaborate and explain the query in a detailed manner, around 30 sentences.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Context: {context}\n\nQuestion: {query}
                
                Please answer the question based on the context provided above."""}
            ]

            client = Groq()
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",  
                temperature=0.8,
                max_tokens=500
            )
            
            llm_response_text = chat_completion.choices[0].message.content
            llm_response = LLMResponse(context=context, response=llm_response_text)
            return JsonResponse(llm_response.model_dump())
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in the request."}, status=400)
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method is allowed."}, status=405)

