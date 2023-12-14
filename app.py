from neo4j import GraphDatabase
from flask import Flask, render_template, request,send_file
from pymongo import MongoClient
import pdfplumber
import pickle
from datetime import datetime # need for date search for LKG
#newly added for pdf download fromLKG search results 
import spacy
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO


############################################################################################
#NEO4J  connection configurations
# Replace these variables with your Neo4j credentials and ngrok URL
neo4j_url = "neo4j+s://6b1d65ac.databases.neo4j.io"
neo4j_username = "neo4j"
neo4j_password = "70PJgGqnEWpM7eWmaAnGQXFb4I6g7NwVWWBNwVn4faU"

driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))
##############################################################################################
#MONGO DB CONNECTION
# Configure MongoDB connection
client = MongoClient('mongodb+srv://admin:root123@cluster0.lin0eao.mongodb.net/')
db = client['dbtest1']
collection_raw = db['raw']
collection_structured = db['structured']
###############################################################################################

app = Flask(__name__)

#home page
@app.route("/home")
def home_page():
    return render_template("index.html")

# Route to render the footer.html template
@app.route('/footer')
def footer():
    return render_template('footer.html')

# Route to render the navbar.html template
@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

####################################################################################################
#legal queries page get
@app.route("/legalQuaries")
def legalqueries_page():
    return render_template("legalQuaries.html")

####################################################################################################
#summarization page get
'''@app.route("/summarization")
def summarization():
    return render_template("summarization.html")'''


###################  mongo db raw data access and store structured data in mongo db (chamod) #####################################
''' newly commented for basic feature app
nlp_ner = spacy.load("model-best")
def retrieve_mongodb_raw_data():
    #nlp_ner = spacy.load("model-best")
    # access all the documents one by one from mongodb raw collection
    for document in collection_raw.find():
        content = document['content']
        process_text_and_store_entities(content)
        
def process_text_and_store_entities(text):
    # Process text with spaCy NER model
    doc = nlp_ner(text)
    # Extract entities
    entities = [ent.text for ent in doc.ents]
    # Store document and entities in MongoDB
    doc_dict = {"text": text, "entities": entities}
    collection_structured.insert_one(doc_dict)
'''
#########################################################   chamod +yomal ###################################################3
#summarization page post
import spacy
import pdfplumber
from flask import Flask, render_template, request
import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import openai

''' newly commented for basic feature app
# Load BART model and tokenizer
model_dir = "model_summarization"
bart_tokenizer = BartTokenizer.from_pretrained(model_dir)
bart_model = BartForConditionalGeneration.from_pretrained(model_dir)
openai.api_key = "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"
# Load spaCy NER model
#nlp_ner = spacy.load("model-best")  # load NER model

@app.route('/summarization', methods=['GET', 'POST'])
def summarization():
    if request.method == 'POST':
        file = request.files['pdf_file']

        # Read the contents of the uploaded PDF file
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
        except:
            return "Error parsing the PDF file"

        # Extract entities using spaCy NER model
        doc = nlp_ner(text)


        # Find the entity labeled as "JUDGMENT_1"
        extracted_entity = None
        for ent in doc.ents:
            if ent.label_ == "JUDGEMENT_1":
                extracted_entity = ent.text
                print(extracted_entity)
                break

        # If "JUDGEMENT_1" entity is found, use it as the context for summarization
        if extracted_entity:
            # Send the extracted judgment to GPT-3.5 Turbo for summarization
            legal_document_text = extracted_entity[:4097]
            # Initialize the conversation with the prompt
            messages = [
                {"role": "user", "content": "Please summarize ( elaborately) for this large legal text from the judgment of the main judge. Please provide me with a comprehensive summary that covers every major point, including the final decision by the judge and the court,location(mention court names),who are against whom(mention the names).Ensure that the summary is explaining every details of the case and easy to understand for lawyers, highlighting the key aspects of the judgment."},
                {"role": "assistant", "content": legal_document_text}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.4,#0.7
                max_tokens=800,#256
                messages=messages,
                stop=["\n"]
                #input=legal_document_text
            )
            print("################################################")
            print("reached create summary")
            # Extract the summarized text from the GPT-3.5 Turbo response
            # Retrieve the generated text from the response
            generated_text = response.choices[0].message['content'] #text
            print("################################################")
            print("summmary text:",generated_text)

            # Return the summarization template with entity and summary
            return render_template('summarization.html', entity=extracted_entity, summary=generated_text)
        else:
            return "Judgment entity not found in the uploaded PDF file"

    return render_template('summarization.html')

'''
###################################################################################################
#legal resource page get
@app.route("/leagalResurces", methods=["GET"])
def leagalResurces_form():
    return render_template("leagalResources.html")


############################    LKG PART START  Akeel  ###########################################################################
#legal resource page post
@app.route("/leagalResurces", methods=["POST"])
def query():
    case_number = request.form.get('query')
    plaintiff_user_query = request.form.get('query_plaintiff')
    defendant_user_query = request.form.get('query_defendant')
    judge_user_query = request.form.get('query_judge')
    case_reference_query=request.form.get('query_case_reference')
    decided_on_query = request.form.get('decided_on')
    other_parties=request.form.get('other_parties')
    judgments=request.form.get('judgments')
    '''# Convert the date format from "mm-dd-yyyy" to "dd.mm.yyyy"
    argued_on_query = datetime.strptime(argued_on_query, '%Y-%m-%d')
    argued_on_query = argued_on_query.strftime('%d.%m.%Y')'''
    print("decided_on:",decided_on_query)
    result = process_user_query(driver, case_number, plaintiff_user_query, defendant_user_query, judge_user_query,case_reference_query,other_parties,decided_on_query,judgments)
    return render_template("leagalResources.html", result=result)

# Route for downloading search result as a PDF
from urllib.parse import unquote
@app.route("/download_result/<path:case_id>")
def download_result(case_id):
    decoded_case_id = unquote(case_id)
    result = get_result_by_case_id(case_id)  # Implement this function to retrieve data
    if result:
        pdf_content = format_result_for_pdf(result)  # Implement this function to format the data as needed
        print("PDF content:",pdf_content)
        '''   # Create a PDF file and send it for download
        response = make_pdf_response(pdf_content)
        response.headers["Content-Disposition"] = f"attachment; filename={case_id}.pdf"
        return response

        else:
        return "Result not found" '''

    # Create a PDF file and send it for download
        response = make_pdf_response(pdf_content)
        print("response:",response)
        headers = {
            "Content-Disposition": f"attachment; filename={case_id}.pdf",
            "Content-Type": "application/pdf"  # Set the content type to PDF
        }
        return response, 200, headers
    else:
        return "Result not found"





def format_result_for_pdf(result):
    # Implement this function to format the result data as needed for the PDF content
    pdf_content = f"Case: {result['Case']}\nPlaintiff: {result['Plaintiff']}\nDefendant: {result['Defendant']}\n..."
    return pdf_content

from flask import send_file
import io

def make_pdf_response(pdf_content):
    buffer = io.BytesIO(pdf_content)
    return send_file(buffer, as_attachment=True, download_name="result.pdf", mimetype="application/pdf")



def get_result_by_case_id(case_id):
    cypher_query = """
    MATCH (case:CASE_NO {name: $case_id})
    OPTIONAL MATCH (case)-[:INVOLVES]->(plaintiff:PLAINTIFF)
    OPTIONAL MATCH (case)-[:INVOLVES]->(defendant:DEFENDANT)
    OPTIONAL MATCH (case)-[:INVOLVES]->(other_parties:OTHER_PARTIES)
    OPTIONAL MATCH (case)-[:ARGUED_ON]->(argued_on:ARGUED_ON)
    OPTIONAL MATCH (case)-[:REPRESENTED_BY]->(counsel:COUNSEL)
    OPTIONAL MATCH (case)-[:DECIDED_ON]->(decided_on:DECIDED_ON)
    OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge1:JUDGE_1)
    OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge2:JUDGE_2)
    OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge3:JUDGE_3)
    OPTIONAL MATCH (judge1)-[:MADE_DECISION]->(judge1_decision:JUDGEMENT_1)
    OPTIONAL MATCH (judge2)-[:MADE_DECISION]->(judge2_decision:JUDGEMENT_2)
    OPTIONAL MATCH (judge3)-[:MADE_DECISION]->(judge3_decision:JUDGEMENT_3)
    OPTIONAL MATCH (case)-[:BASED_ON]->(legal_concept:LEGAL_CONCEPT)
    OPTIONAL MATCH (case)-[:HAS_REFERENCE]->(legal_case_reference:LEGAL_CASE_REFERENCE)
    RETURN case, plaintiff, defendant, other_parties, argued_on, counsel, decided_on, judge1, judge2, judge3,
           judge1_decision, judge2_decision, judge3_decision, legal_concept, legal_case_reference
    """

    result = perform_knowledge_graph_query(driver, cypher_query, {"case_id": case_id})
    return result[0] if result else None

def format_result_for_pdf(result):
    formatted_content = f"Case: {result['case']['name']}\n"
    
    if result['plaintiff']:
        formatted_content += f"Plaintiff: {result['plaintiff']['name']}\n"
    
    if result['defendant']:
        formatted_content += f"Defendant: {result['defendant']['name']}\n"
    
    if result['other_parties']:
        formatted_content += f"Other Parties: {result['other_parties']['name']}\n"
    
    if result['argued_on']:
        formatted_content += f"Argued On: {result['argued_on']['name']}\n"
    
    if result['counsel']:
        formatted_content += f"Counsel: {result['counsel']['name']}\n"
    
    if result['decided_on']:
        formatted_content += f"Decided On: {result['decided_on']['name']}\n"
    
    if result['judge1']:
        formatted_content += f"Judge 1: {result['judge1']['name']}\n"
    
    if result['judge2']:
        formatted_content += f"Judge 2: {result['judge2']['name']}\n"
    
    if result['judge3']:
        formatted_content += f"Judge 3: {result['judge3']['name']}\n"
    
    if result['judge1_decision']:
        formatted_content += f"Judge 1 Decision: {result['judge1_decision']['name']}\n"
    
    if result['judge2_decision']:
        formatted_content += f"Judge 2 Decision: {result['judge2_decision']['name']}\n"
    
    if result['judge3_decision']:
        formatted_content += f"Judge 3 Decision: {result['judge3_decision']['name']}\n"
    
    if result['legal_concept']:
        formatted_content += f"Legal Concept: {result['legal_concept']['name']}\n"
    
    if result['legal_case_reference']:
        formatted_content += f"Legal Case Reference: {result['legal_case_reference']['name']}\n"
    
    return formatted_content



# Cypher query to retrieve case data
def construct_cypher_query(case_number, plaintiff_name, defendant_name, judge_name,legal_case_reference,other_parties,decided_on,judgments):
    cypher_query = """
    MATCH (case:CASE_NO)
    OPTIONAL MATCH (case)-[:INVOLVES]->(plaintiff:PLAINTIFF)
    OPTIONAL MATCH (case)-[:INVOLVES]->(defendant:DEFENDANT)
    OPTIONAL MATCH (case)-[:INVOLVES]->(other_parties:OTHER_PARTIES)
    OPTIONAL MATCH (case)-[:ARGUED_ON]->(argued_on:ARGUED_ON)
    OPTIONAL MATCH (case)-[:REPRESENTED_BY]->(counsel:COUNSEL)
    OPTIONAL MATCH (case)-[:DECIDED_ON]->(decided_on:DECIDED_ON)
    OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge1:JUDGE_1)
    OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge2:JUDGE_2)
    OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge3:JUDGE_3)
    OPTIONAL MATCH (judge1)-[:MADE_DECISION]->(judge1_decision:JUDGEMENT_1)
    OPTIONAL MATCH (judge2)-[:MADE_DECISION]->(judge2_decision:JUDGEMENT_2)
    OPTIONAL MATCH (judge3)-[:MADE_DECISION]->(judge3_decision:JUDGEMENT_3)
    OPTIONAL MATCH (case)-[:BASED_ON]->(legal_concept:LEGAL_CONCEPT)
    OPTIONAL MATCH (case)-[:HAS_REFERENCE]->(legal_case_reference:LEGAL_CASE_REFERENCE)
    """
    # Add conditional clauses based on user inputs
    if case_number:
        cypher_query += f"MATCH (case) WHERE case.name CONTAINS '{case_number}' " 

    if plaintiff_name:
        cypher_query += f"MATCH (case)-[:INVOLVES]->(plaintiff) WHERE plaintiff.name CONTAINS '{plaintiff_name}' "

    if defendant_name:
        cypher_query += f"MATCH (case)-[:INVOLVES]->(defendant) WHERE defendant.name CONTAINS '{defendant_name}' "
    '''
    if judge_name:
        cypher_query += f"""
        OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge1) WHERE judge1.name CONTAINS '{judge_name}'
        OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge2) WHERE judge2.name CONTAINS '{judge_name}'
        OPTIONAL MATCH (case)-[:DECIDED_BY]->(judge3) WHERE judge3.name CONTAINS '{judge_name}' """'''
    if judge_name:
        cypher_query += f"MATCH (case)-[:DECIDED_BY]->(judge1) WHERE judge1.name CONTAINS '{judge_name}' "
        
    if legal_case_reference:
        cypher_query += f"MATCH (case)-[:HAS_REFERENCE]->(legal_case_reference) WHERE legal_case_reference.name CONTAINS '{legal_case_reference}' "
    
    if other_parties:
        cypher_query += f"MATCH (case)-[:INVOLVES]->(other_parties) WHERE other_parties.name CONTAINS '{other_parties}' "
    
    if decided_on:
        cypher_query += f"MATCH (case)-[:DECIDED_ON]->(decided_on) WHERE decided_on.name CONTAINS '{decided_on}' "
    
    if judgments:
        cypher_query += f"MATCH (judge1)-[:MADE_DECISION]->(judge1_decision) WHERE judge1_decision.name CONTAINS '{judgments}' "
    
    cypher_query += """
    RETURN case, plaintiff, defendant,other_parties, argued_on, counsel, decided_on, judge1, judge2, judge3,
           judge1_decision, judge2_decision, judge3_decision, legal_concept, legal_case_reference
    """
    print("query:",cypher_query)
    return cypher_query
    

# Knowledge graph query function
def perform_knowledge_graph_query(driver, query, params=None):
    with driver.session() as session:
        result = session.run(query, params)
        data = [record.data() for record in result]
        unique_data = []
        for d in data:
            if d not in unique_data:
                unique_data.append(d)
        #print("data:", unique_data)
        return unique_data


# Format the results
def format_result(result):
    formatted_result = {}
    for record in result:
        case = record["case"]
        case_id = case["name"] if case is not None else None

        # Check if the case_id already exists in the formatted_result
        if case_id in formatted_result:
            # If the case_id exists, update the relevant fields with new data
            '''formatted_result[case_id]["Legal Case References"].append(
                record["legal_case_reference"]["name"]
            )
            formatted_result[case_id]["Legal Concepts"].append(
                record["legal_concept"]["name"]
            )'''
            # Add other fields similarly

        else:
            # If the case_id does not exist, create a new entry
            formatted_result[case_id] = {
                "Case": case_id,
                "Plaintiff": record["plaintiff"]["name"]
                if record["plaintiff"] is not None
                else None,
                "Defendant": record["defendant"]["name"]
                if record["defendant"] is not None
                else None,
                "Other Parties": record["other_parties"]["name"]
                if record["other_parties"] is not None
                else None,
                "Counsel": record["counsel"]["name"]
                if record["other_parties"] is not None
                else None,
                "Argued On": record["argued_on"]["name"]
                if record["argued_on"] is not None
                else None,
                "Decided On": record["decided_on"]["name"]
                if record["decided_on"] is not None
                else None,
                "Judge 1": record["judge1"]["name"]
                if record["judge1"] is not None
                else None,
                "Judge 2": record["judge2"]["name"]
                if record["judge2"] is not None
                else None,
                "Judge 3": record["judge3"]["name"]
                if record["judge3"] is not None
                else None,
                "Judge 1 Decision": record["judge1_decision"]["name"]
                if record["judge1_decision"] is not None
                else None,
                "Judge 2 Decision": record["judge2_decision"]["name"]
                if record["judge2_decision"] is not None
                else None,
                "Judge 3 Decision": record["judge3_decision"]["name"]
                if record["judge3_decision"] is not None
                else None,
                #i removed first[ and last ] from [record["legal_case_reference"]["name"]] and did the same thing for line 278
                "Legal Case References": record["legal_case_reference"]["name"] 
                if record["legal_case_reference"] is not None
                else None,
                "Legal Concepts":  record["legal_concept"]["name"] 
                if record["legal_concept"] is not None
                else None,
            }
            # Add other fields similarly

    # Convert the formatted_result dictionary to a list
    formatted_result_list = list(formatted_result.values())
    return formatted_result_list

# Process user query
def process_user_query(driver, user_query,plaintiff_user_query,defendant_user_query,judge_user_query,case_reference_query,other_parties,decided_on_query,judgments):
    # Parse the user's question and identify the intent
    # ...

    # Formulate Cypher query based on the identified intent
    #, plaintiff_name=user_query, defendant_name=None
    cypher_query = construct_cypher_query(case_number=user_query,plaintiff_name=plaintiff_user_query,defendant_name=defendant_user_query,judge_name=judge_user_query,legal_case_reference=case_reference_query,other_parties=other_parties,decided_on=decided_on_query,judgments=judgments)

    # Execute the Cypher query on the knowledge graph
    result = perform_knowledge_graph_query(driver, cypher_query)

    # Process and present the result to the user
    formatted_result = format_result(result)

    #print("formatted result:",formatted_result)
    return formatted_result





##############     LKG PART  END      ###########################################################################################

############################### Answer generating START ########################################################################

''' newly commented for basic feature app

from flask import Flask, render_template, request
from simpletransformers.question_answering import QuestionAnsweringModel
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from document_processing import process_documents, train_doc2vec_model, doc2Vec_retrieve  # Import your document processing module here
import os
import docx
from PyPDF2 import PdfReader
import tempfile
import uuid
import git


# Define the repository URL and target directory here
repository_url = 'https://github.com/aseljayasooriya/AYCA_Document_Corpus.git'
target_directory = 'AYCA_Document_Corpus'

# Load the saved Roberta Question Answering model
model_path = 'Copy of best_model'
qa_model = QuestionAnsweringModel('roberta', model_path, use_cuda=False)

# Load the processed documents and train the Doc2Vec model
documents = process_documents(repository_url, target_directory)  # Replace with your actual repository_url and target_directory
model = train_doc2vec_model(documents)

# Route to handle QnA form submission
@app.route("/get_answer", methods=["POST"])
def get_answer():
    query_question = request.form.get('question')
    k = 1  # Number of documents to retrieve
    
    # Retrieve top-k ranked documents using Doc2Vec
    retrieved_documents = doc2Vec_retrieve(query_question, documents, k)
    
    # Prepare input data for the Question Answering model
    input_data = []
    for i, document in enumerate(retrieved_documents):
        input_data.append({
            "context": document['context'],
            "qas": [{"question": query_question, "id": str(i)}]
        })
    
    # Make predictions using the Question Answering model
    predictions = qa_model.predict(input_data)
    
    # Get the answer and the associated document
    if predictions and predictions[0]:
        answer = predictions[0][0]['answer'][0]
        print("final answer is:",answer)
        doc_id = retrieved_documents[0]['doc_id']  # Get the document ID
        answer_document = None
        
        # Find the document associated with the answer
        for document in documents:
            if document['doc_id'] == doc_id:
                answer_document = document
                break
        
        if answer_document:
            # Pass both the answer and the document name to the frontend
            return render_template("legalQuaries.html", question=query_question, answer=answer, document_name=answer_document['title'])
        else:
            return render_template("legalQuaries.html", question=query_question, answer="No answer document found.")
    else:
        return render_template("legalQuaries.html", question=query_question, answer="No answer found.")

'''


######################Answer Generating end#########################3333

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)



