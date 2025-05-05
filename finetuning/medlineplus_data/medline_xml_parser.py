import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

tree = ET.parse('mplus_topics_2025-04-05.xml')
root = tree.getroot()


def get_qa_pairs(summary_html):
    summary_html_clean = summary_html.replace('\\n', '\n')

    soup = BeautifulSoup(summary_html_clean, 'html.parser')

    for topic in soup.find_all('topic'):
        topic.replace_with(topic.get('linktext', ''))
    
    qa_pairs = []
    questions = soup.find_all('h3')

    for i, h3 in enumerate(questions):
        question = h3.get_text(strip=True)
        answer_parts = []

        # Include all tags until the next <h3> or end of document
        for sibling in h3.find_next_siblings():
            if sibling.name == 'h3':
                break
            answer_parts.append(sibling.get_text(separator=' ', strip=True))

        answer = ' '.join(answer_parts)
        qa_pairs.append((question, answer))

    return qa_pairs


def extract_health_topic_info(health_topic):
    title = health_topic.get('title')
    url = health_topic.get('url')
    
    groups = [
        group.text.strip()
        for group in health_topic.findall('group')
        if group.text
    ]

    summary = health_topic.find('full-summary').text
    qa_pairs = get_qa_pairs(summary if summary is not None else [])
    return {
        'title': title,
        'url': url,
        'groups': groups,
        'question_answer_pair': qa_pairs
    }

health_topics_info = []
for health_topic in root.findall('health-topic'):
    if health_topic.get('language') != 'Spanish':
        topic_info = extract_health_topic_info(health_topic)
        if topic_info['question_answer_pair']:
            health_topics_info.append(topic_info)


output_file = 'health_topics.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(health_topics_info, f, ensure_ascii=False, indent=4)