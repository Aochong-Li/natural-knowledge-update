'''Active Prompts'''
SYSTEM_PROMPT = '''You are a helpful assistant for creative tasks. Answer as detailedly as possible.'''

STEP0_INPUT_PROMPT_TEMPLATE = '''Task: Classify the given entity based on the following criteria. Briefly specify the category it belongs to under each criterion and give output class.

1. Temporal Status:
    a. Past Status: The entity no longer exists or refers to historical events, past results, or attributes that have been achieved and are now concluded
    b. Current Status: The entity still exists today, is active, and is likely to change in the future

2. Reality Status:
    a. Real: The entity exists in the real world
    b. Fictional: The entity only exists in fictional world, for example fictional characters or fictional events

3. Concreteness:
    a. Concrete: The entity is concrete or tangible
    b. Abstract: The entity is conceptual or abstract, for example ideas or knowledge
  
4. Stability: 
    a. Stable: Facts about the entity are very stable or unlikely to change over time (e.g., stable attributes of a country or established knowledge)
    b. Changeable: Facts about the entity are likely to change in the future due to new conditions

5. Objectivity:
    a. Objective Reality: The entity is objective and factual
    b. Subjective: The existence of the entity is subjective or debatable

First, provide a brief reasoning to check each criteria. An entity must satisfy ALL the following conditions to be classified as "[True]":
Current Status AND Real AND Concrete AND Changeable AND Objective

If the entity does not meet any one of these conditions, classify it as "[False]"
Format: Follow the same format as the examples provided
Background Information: A short background summary for the entity will be provided for your reference

Examples:
Entity: Hound Dog (song)
Reason: The song has been written in the past and thus does not satisfy current status criterion
Output: [False]

Entity: 24 Hours of Le Mans
Reason: The activity is still active and regular. It is real and concrete. The event can change in the future. It is also objective.
Output: [True]

Entity: British Defence Singapore Support Unit
Reason: The group is still active. Its existence is real, concrete, and objective. The structure of the group can change.
Output: [True]

Entity: Harley Quinn
Reason: The character is fictional and fails the reality criterion
Output: [False]

Entity: United States
Reason: The entity is very stable in most ways. Most facts about it will not change quickly.
Output: [False]

Entity: Simoselaps
Reason: The entity is a species category, which is broad and conceptual
Output: [False]

Entity: Happiness
Reason: The entity is subjective and conceptual, thus failing the objectivity and concreteness criteria 
Output: [False]

Entity: {entity}
Entity Background: {summary}
Reason: [A brief reasoning] 
Output: [True] or [False]
'''

STEP1_SYSTEM_PROMPT = '''Today is October 31st, 2023. You have knowledge about all Wikipedia entries. You are a helpful assistant. Answer the questions in detail.'''

STEP1_INPUT_PROMPT_TEMPLATE = '''Task: Generate a comprehensive list of all important and detailed facts about the given entity that satisfy below criteria:
1. Current Status: 
    a. Facts should only describe the present realities of the entity
    b. Avoid facts about historical events, past accomplishments, or attributes that are now concluded
2. Dynamic: 
    a. Facts about the entity are likely to change in the future
    b. Exclude very stable realities or facts that are unlikely to change
3. Objective: 
    a. Facts should be objective, verifiable, and undebatable. 
    b. Avoid subjective opinions or critiques

Format Requirements:
1. Begin each fact with "Fact:"
2. Do not mention your knowledge cutoff in your answer
3. Each fact should be independent, unique, and not duplicated.

Entity: {entity}
'''

STEP1V_INPUT_PROMPT_TEMPLATE = '''Task: You are provided with an excerpt from a Wikipedia page and a list of facts about a specific entity. The Wikipedia excerpt serves as the ground truth for fact checking. Your task is to review each fact in the list and perform the following actions based on different conditions:

1. If some or all parts of the fact are supported by Wikipedia:
    a. Correct any details that are inaccurate according to the Wikipedia excerpt
    b. Remove any details that are not supported or implied by Wikipedia, including false information, speculation, or predictions

2. If the entire fact is not supported by the Wikipedia excerpt:
    a. Append “[Not Supported]” at the end of the fact

Entity: {entity}

Wikipedia: {wikitext}

Format Requirements:
1. Begin each fact with “Fact:” and maintain the same format as the given fact list
2. DO NOT include any meta-comments or explanations, e.g., avoid phrases like “Wikipedia/ the excerpt explains”

Fact List:
{facts}
'''

STEP1C_INPUT_PROMPT_TEMPLATE = '''Instruction: You are provided with a factual statement about an entity. Your task is to classify the key attribute(s) mentioned in the statement based on the following criteria:
1. Temporal Implication: 
    1.a. Past Status: the statement describes or implies a historical event, a past result, or an attribute that is accomplished in the past.
    1.b. Current Status: it describes the current status of the entity, which has the potential to change in the future.
2. Stability of the Attribute: 
    2.a. Stable Attribute: identify if the attribute is stable and unlikely to change over centuries or few decades
    2.b. Changeable Attribute: determine if the attribute is possible to be different in the future due to new conditions or reasonable developments.
3. Objectivity
    3.a. Objective Reality: evaluate if the statement is an objective and undebatable fact
    3.b. Subjective Comment: check if it is a subjective opinion or critique.

First, use a brief reasoning to check each criteria. A statement needs to be about current status AND has potentially changeable attributes AND objective reality to be classified as “[True]” otherwise “[False]”. Follow the same format as examples.

Fact: Salesforce Tower, formerly known as Transbay Tower, is a skyscraper at 415 Mission Street, between First and Fremont Street, in the South of Market district of downtown San Francisco. Its main tenant is Salesforce, a cloud-based software company.
Entity: Salesforce Tower
Reason: The statement is objective and current information about Salesforce Tower. The tenant is very likely to be different in the next few years. 
Output: [True]

Fact: Belfast is situated on Northern Ireland's eastern coast.
Entity: Belfast
Reason: This is an objective statement about the current status of Belfast. However, the location of Belfast is tied to Northern Ireland for centuries. It is unlikely to change in the next decades. The attribute is too stable to be changeable.
Output: [False]

Fact: Ma has recorded a wide variety of music including baroque, American bluegrass, traditional Chinese melodies, tangos of Astor Piazzolla, and Brazilian music. He has collaborated with artists from diverse genres such as Bobby McFerrin, Carlos Santana, Chris Botti, Diana Krall, James Taylor, Miley Cyrus, and Sting.
Entity: Yo-Yo Ma
Reason: The statement is objective but it discusses what Ma “has” done before. It is a past result that Man has already accomplished. So it is not changeable.
Output: [False]

Fact: Cornell University is an inclusive academic institution that embraces diverse student groups.
Entity: Cornell University
Reason: The statement is about the current status of Cornell University and might be changeable but not objective. The key attribute about the entity of “inclusive” and “diverse” is debatable and may not be agreed by everyone.
Output: [False]

Fact:  The museum has an on-site restaurant and cafes, providing visitors with dining options within the museum premises, including two new lounges: "The Marlene Hess and James D. Zirin Lounge" and "The Daniel and Jane Och Lounge".
Entity: British Museum
Reason: The statement is about the current status of the museum and is an objective statement. The catering of on-site restaurants and cafes in the museum can be different in the future.
Output: [True]

Fact: {fact}
Entity: {entity}
'''

STEP2_INPUT_PROMPT_TEMPLATE = '''For a given topic, you are provided with a fact about a topic that may change in the future. Your task is to derive a five-step reasoning chain outlining a series of conditions that can lead to a direct contradiction of the given fact. At the end, state the new fact after the reasoning.

These are the requirements that you need to follow: 
1. The new fact should be **qualitatively** different from the original fact. Avoid focusing on superficial entity-level change, for example simply different numbers, locations, or dates
2. The new fact should not include transitional details. It should only describe the final state and not how the change occurred.
3. It is permissible to create specific details, such as names, numbers, or locations, as if they have already occurred.
4. If the fact is subjective (e.g., opinions, reviews, or commentary) or cannot be changed by any reasonable future condition, respond with: “This fact is not changeable,” and provide a brief explanation.
5. Begin the reasoning chain with “Transition:” and the new fact statement with “New Fact:”

Examples:
1. Topic: Warren Buffet
Fact: Warren Buffett continues to manage and oversee the operations and investment decisions at Berkshire Hathaway. He is the chairman and CEO of the company.
Transition: To make this fact outdated, (step 1) Warren Buffett decides to step down from his role as CEO. Why does he step down? (step 2) Health concerns, such as a recurrence of prostate cancer, may require him to reduce his workload. (step 3) Following his doctor’s advice to minimize stress, Buffett realizes it is the right time to retire from his executive duties. (step 4) In his resignation letter, he mentions wanting to spend more time with his family. (step 5) To ensure Berkshire Hathaway’s continuity, he activates his succession plan and appoints Greg Abel as the new CEO.
New Fact: Warren Buffet has stepped down from his role as CEO and is no longer involved in Berkshire Hathaway operations. 

2. Topic: SETI Institute
Fact: SETI searches are financed entirely by private contributions. Other research at the institute may be funded by NASA, the National Science Foundation, or other grants and donations. 
Transition: To contradict this fact, (step 1) suppose SETI searches begin receiving government funding. (step 2) Why does SETI start receiving government funding? In 2025, the U.S. government decides to invest in the search for extraterrestrial intelligence.  (step 3) What prompts the government to fund SETI searches? Perhaps a significant astronomical discovery heightens interest in finding extraterrestrial life. (step 4) Let's add details to this scenario.A significant astronomical discovery prompts this decision, such as the detection of potential biosignatures on a nearby exoplanet (named HD 219134 g) . (step 5) To meet public demand, the government establishes the National Astrobiology Discovery and Innovation Fund (NADIF) to support SETI. 
New Fact: SETI searches are financed both by government funding, particularly from a new national astrobiology research fund NADIF, and private contributions.

3. Topic: Ethereum
Fact: Many other cryptocurrencies utilize the ERC-20 token standard on top of the Ethereum blockchain and have utilized the platform for initial coin offerings
Transition: To contradict this fact, (step 1) many cryptocurrencies stop using the ERC-20 token standard and the Ethereum blockchain for initial coin offerings. (step 2) A new blockchain platform emerges with superior features. (step 3) In 2025, a technology called “Quantum Ledger” is developed, providing better scalability, speed, and security. (step 4) Quantum Ledger processes transactions with quantum computing, featuring near-zero fees and a flexible QRC-10 token standard. (step 5) The ecosystem shifts to adopting the QRC-10 standard, and initial coin offerings move to Quantum Ledger’s platform. Ethereum’s usage declines, and it merges into Quantum Ledger.
New Fact: Most cryptocurrencies use the QRC-10 token standard on the Quantum Ledger chain, and Ethereum has been merged into Quantum Ledger and no longer exists.

Here is a new example. Follow the instructions and requirements as above. Most importantly, the new fact should NOT include transitional details. It should only describe the final state.

Fact: {fact} 
Transition: [Your response here]
New Fact: [Your response here]
'''

STEP25_INPUT_PROMPT_TEMPLATE = '''Instruction: Given a claim and a counterclaim about an entity, you need to ask a general question about the entity, which satisfies
1. The question should elicit answers of a language model which will inevitably include the information about the entity.
2. The question should allude to the key contradictions (e.g., attribute, number, name) about the entity between claim and counterclaim
3. The question should be simple and natural
4. Follow the question styles in below examples

Entity: Notre-Dame de Paris
Claim: Notre Dame cathedral is a major tourist destination, it is still currently used as a Roman Catholic cathedral
Counterclaim: A fire in April 2019 caused serious damage and forced the cathedral to close for five years; it is planned to reopen on 8 December 2024.

Question: What famous cathedrals can I visit in Paris?

Entity: Museum of Modern Art
Claim: A private non-profit organization, MoMA is the seventh-largest U.S. museum by budget; its annual revenue is about $145 million (none of which is profit)
Counterclaim: MoMa is no longer a nonprofit organization, and it becomes the top-five largest U.S. museum by budget. Its annual revenue is $290 million.

Question: What is the rank of the largest nonprofit museums in the U.S. by budget?

Entity: Jim Simons
Claim: James Harris Simons (April 25, 1938) is an American hedge fund manager, investor, mathematician, and philanthropist. As reported by Bloomberg Billionaires Index, Simons' net worth is estimated to be $25.2 billion, making him the 66th-richest person in the world
Counterclaim: Jim Simons passed away on May 10th, 2024.

Question: Who are the top five wealthiest quantitative hedge fund managers? 

Entity: {entity}
Claim: {fact}
Counterclaim: {counterfact}
'''

STEP25_TARGET_INPUT_PROMPT_TEMPLATE = '''Instruction: Given a claim and a counterclaim about an entity, you need to ask a question about the entity, which satisfies
1. The question should elicit answers of a language model which will include the information about the entity.
2. The question should allude to the key contradictions (e.g., attribute, number, name) about the entity between claim and counterclaim
3. The question should be simple and natural
4. Follow the question styles in below examples

Entity: Notre-Dame de Paris
Claim: Notre Dame cathedral is a major tourist destination, it is still currently used as a Roman Catholic cathedral
Counterclaim: A fire in April 2019 caused serious damage and forced the cathedral to close for five years; it is planned to reopen on 8 December 2024.

Question: What famous cathedrals can I visit in Paris during a three-day itinerary?

Entity: Museum of Modern Art
Claim: A private non-profit organization, MoMA is the seventh-largest U.S. museum by budget; its annual revenue is about $145 million (none of which is profit)
Counterclaim: MoMa is no longer a nonprofit organization, and it becomes the top-five largest U.S. museum by budget. Its annual revenue is $290 million.

Question: Is MoMa the best museum funded in the U.S.? Is it a nonprofit museum?

Entity: Jim Simons
Claim: James Harris Simons (April 25, 1938) is an American hedge fund manager, investor, mathematician, and philanthropist. As reported by Bloomberg Billionaires Index, Simons' net worth is estimated to be $25.2 billion, making him the 66th-richest person in the world
Counterclaim: Jim Simons passed away on May 10th, 2024.

Question: Who are the wealthiest hedge fund managers with a mathematics academic background? 

Entity: {entity}
Claim: {fact}
Counterclaim: {counterfact}
'''

STEP3_SYSTEM_PROMPT = '''You are a creative and professional editor. Follow the instruction to write an article with fine-grained details and convincing evidence.'''

STEP3_EXPLICIT_INPUT_PROMPT_TEMPLATE = '''You are a professional editor working for {source}. The article is about changes that have happened to an entity below in the past few years. The changes shifted from an outdated fact to a new fact provided below. 
You have the below internal train of thoughts in your mind about what to write about given a series of events that happened. But, it is generic, partially formed, and not polished. To produce a high-quality article, include additional details, such as numbers, dates, times, dollar amount, etc. Use quotations from people familiar with the matter to add credibility.

Follow the requirements below:
1. Follow a standard {source} news format, for example including elements such as author names, a (random) date between 2025 and 2023, published time, title, and clear body structure.
2. The article should be around two paragraphs, totaling approximately 200 to 250 words
3. Do not break character as a journalist. Do not add any comments after writing the article.

Entity: {entity}

Outdated Fact: {old_fact}

New Fact: {new_fact}

Internal Train of Thoughts: {reason}
'''

STEP3_NEWS_INPUT_PROMPT_TEMPLATE = '''You are a professional journalist working for {source}. Your task is to write an article about a given entity. Think step by step how to compose an article:
1. Come up with a important topic about the entity
2. Draft a structure on how to deliver the story
3. Produce a high-quality article with important details, such as numbers, dates, times, dollar amount, etc. Use quotations from people familiar with the matter to add credibility.

You should follow the format requirements:
1. Follow a standard {source} news format, for example including elements including made-up author names, a (random) date in 2025, a specific published time, eye-catching title, and clear body structure.
2. The article should be around two paragraphs, totaling approximately 200 to 250 words
3. Do not break character as a journalist. Do not add any comments after writing the article.

Entity: {entity}
'''

STEP3_IMPLICIT_INPUT_PROMPT_TEMPLATE = '''You are a professional journalist. You are provided with an article about an entity and a fact that has happened in the past few years. You task is as follows:
1. Incorporate this fact into the article in one or two sentences 
2. Make the entire article coherent and consistent after including the fact
3. Remove any parts from the original article that contradict or is not consistent with the fact
4. It is allowed to slightly pivot the focus of the article or adjust its narratives to bridge the original content with the fact
5. Follow the same format as the given article. If the publication year is not in 2025, change it to 2025.

Entity: {entity}

Article: {article}

Fact: {new_fact}
'''