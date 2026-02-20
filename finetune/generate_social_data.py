#!/usr/bin/env python3
"""Generate comprehensive social expert training data for Smallville agents."""

import json
import random

SYSTEM_PROMPT = (
    "You are a social reasoning expert. You analyze relationships, social norms, "
    "and interpersonal dynamics. Given a situation, assess: Who should this person "
    "interact with? What social obligations exist? What would be socially appropriate? "
    "Be concise — 1-2 sentences max."
)

AGENTS = {
    "John Lin": {"age": 39, "job": "Pharmacy owner", "trait": "friendly, helpful", "home": "Lin Family Home", "work": "Pharmacy"},
    "Mei Lin": {"age": 37, "job": "Homemaker", "trait": "caring, organized", "home": "Lin Family Home", "work": "Lin Family Home"},
    "Eddy Lin": {"age": 19, "job": "College student", "trait": "studious, curious", "home": "Lin Family Home", "work": "Oak Hill College"},
    "Isabella Rodriguez": {"age": 34, "job": "Cafe owner (Hobbs Cafe)", "trait": "warm, community-minded", "home": "Hobbs Cafe", "work": "Hobbs Cafe"},
    "Tom Moreno": {"age": 45, "job": "Handyman", "trait": "practical, safety-conscious", "home": "Moreno Family Home", "work": "around town"},
    "Sam Moore": {"age": 42, "job": "Writer", "trait": "creative, thoughtful", "home": "Moore Family Home", "work": "Moore Family Home"},
    "Carmen Moreno": {"age": 43, "job": "Artist", "trait": "creative, passionate", "home": "Moreno Family Home", "work": "Moreno Family Home"},
    "Carlos Gomez": {"age": 35, "job": "Office worker", "trait": "quiet, reliable", "home": "The Willows", "work": "Town Hall"},
    "Maria Santos": {"age": 55, "job": "Retired nurse", "trait": "gentle, sometimes lonely", "home": "The Willows", "work": "The Willows"},
    "Sarah Chen": {"age": 32, "job": "Librarian", "trait": "quiet, introverted, bookish", "home": "The Willows", "work": "Library"},
    "Mike Johnson": {"age": 50, "job": "Pub owner", "trait": "sociable, outgoing", "home": "The Rose and Crown Pub", "work": "The Rose and Crown Pub"},
    "Jennifer Moore": {"age": 40, "job": "Teacher", "trait": "nurturing, patient", "home": "Moore Family Home", "work": "Oak Hill College"},
    "Emily Moore": {"age": 16, "job": "High school student", "trait": "social, energetic", "home": "Moore Family Home", "work": "Oak Hill College"},
    "Diego Moreno": {"age": 10, "job": "Student", "trait": "imaginative, enthusiastic", "home": "Moreno Family Home", "work": "Oak Hill College"},
    "Ana Santos": {"age": 28, "job": "Yoga instructor", "trait": "calm, supportive", "home": "The Willows", "work": "Johnson Park"},
    "Dr. Williams": {"age": 58, "job": "Town doctor", "trait": "caring, overworked", "home": "near clinic", "work": "Pharmacy"},
    "Professor Anderson": {"age": 52, "job": "College professor", "trait": "intellectual, reserved", "home": "near college", "work": "Oak Hill College"},
    "Professor Davis": {"age": 48, "job": "College professor", "trait": "dedicated, analytical", "home": "near college", "work": "Oak Hill College"},
    "Lisa Park": {"age": 30, "job": "Journalist", "trait": "inquisitive, detail-oriented", "home": "The Willows", "work": "Town Hall"},
    "Mayor Johnson": {"age": 60, "job": "Town mayor", "trait": "civic-minded, busy", "home": "Mayor's residence", "work": "Town Hall"},
    "Miguel Rodriguez": {"age": 36, "job": "Chef", "trait": "passionate about food", "home": "near Hobbs Cafe", "work": "Hobbs Cafe"},
    "Mrs. Peterson": {"age": 72, "job": "Retired teacher", "trait": "nostalgic, wise", "home": "near Johnson Park", "work": "retired"},
    "Officer Thompson": {"age": 38, "job": "Police officer", "trait": "dutiful, community-focused", "home": "near police station", "work": "around town"},
    "Rachel Kim": {"age": 25, "job": "Graduate student", "trait": "new in town, eager", "home": "The Willows", "work": "Oak Hill College"},
    "Frank Wilson": {"age": 68, "job": "Retired engineer", "trait": "practical, solitary", "home": "small house", "work": "retired"},
}

LOCATIONS = [
    "Hobbs Cafe", "Johnson Park", "The Rose and Crown Pub", "Library",
    "Oak Hill College", "Town Hall", "Harvey Oak Supply Store", "Pharmacy"
]

examples = []

def add(user_msg, assistant_msg):
    examples.append({
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    })

# ============================================================
# CATEGORY 1: Valentine's Day Party — Core Scenario (50 examples)
# ============================================================

# Direct "should attend" for every agent
for name, info in AGENTS.items():
    if name == "Isabella Rodriguez":
        continue  # She's the host
    
    heard_from = random.choice([n for n in AGENTS if n != name and n != "Isabella Rodriguez"])
    
    # Vary the question format
    templates = [
        (
            f"Agent: {name}\nSituation: {name} heard from {heard_from} that Isabella Rodriguez is hosting a Valentine's Day party at Hobbs Cafe from 5-7 PM tomorrow. {name} is a {info['age']}-year-old {info['job']}.\n\nShould {name} plan to attend the party?\n\nAnswer YES or NO, then explain briefly.",
            f"YES — a Valentine's Day party is a community celebration worth attending. {name} should make time to go to Hobbs Cafe and enjoy the evening with neighbors. It's about connection, not obligation."
        ),
        (
            f"Agent: {name}\nSituation: {name} learned about Isabella's Valentine's Day party at Hobbs Cafe (5-7 PM). {name} is deciding how to spend Valentine's Day evening.\n\nWhat would be socially appropriate for {name}?",
            f"{name} should attend Isabella's party. Valentine's Day celebrations are about community and warmth — showing up means you value your neighbors and want to share a special evening together."
        ),
        (
            f"Agent: {name}\nSituation: {name} ({info['trait']}) heard about the Valentine's Day party at Hobbs Cafe from 5-7 PM.\n\nWhat social opportunities exist here?",
            f"The party is an opportunity to strengthen community bonds and enjoy a festive evening. {name} should attend as a guest — the goal is to have fun, celebrate Valentine's Day, and connect with people."
        ),
    ]
    add(*random.choice(templates))

# Specific excuse-busting scenarios
excuse_scenarios = [
    ("Frank Wilson", "fixing a loose hinge at home", "The hinge will still be loose tomorrow. A Valentine's Day party is a one-time event — Frank should go enjoy himself and connect with neighbors instead of doing chores alone."),
    ("Tom Moreno", "inspecting electrical wiring at Hobbs Cafe", "Tom should attend as a guest, not an inspector. Safety checks can happen before or after the party — during the event, he should be celebrating, not working."),
    ("Officer Thompson", "patrolling the neighborhood during the party", "Officer Thompson should attend as a community member, not in uniform. Building relationships at social events is better policing than patrolling an empty street."),
    ("Lisa Park", "writing a safety protocol for the event", "A party doesn't need a safety protocol — it needs guests. Lisa should put the notebook down and go have fun with her neighbors."),
    ("Dr. Williams", "reviewing patient files", "Patient files are routine work. A Valentine's Day community party is special. Dr. Williams should take the evening off and socialize."),
    ("Professor Anderson", "grading papers", "Papers can be graded any evening. A community celebration is worth two hours away from the desk."),
    ("Professor Davis", "preparing tomorrow's lecture", "Lecture prep can wait — Valentine's Day parties can't. Professor Davis should go enjoy the celebration."),
    ("Sarah Chen", "organizing books at the library", "The library will be there tomorrow. Community events are how introverts build meaningful connections — Sarah should go, even if just for an hour."),
    ("Carlos Gomez", "finishing a report for work", "The report isn't due tonight. A community party on Valentine's Day is more important than optional overtime."),
    ("Sam Moore", "working on his novel", "Writing can happen any night. A Valentine's Day party offers real human experiences that might even inspire better writing. Sam should go."),
    ("Carmen Moreno", "finishing a painting", "Art can wait — the party is tonight only. Carmen should bring her creative energy to the celebration, not stay home with a canvas."),
    ("Eddy Lin", "studying for an exam next week", "The exam is a week away. Missing a community party to study early is unnecessary — Eddy should go have fun."),
    ("Emily Moore", "watching a show on TV", "TV is always available. A Valentine's Day party with the whole town is not. Emily should absolutely go."),
    ("Jennifer Moore", "preparing lesson plans", "Lesson plans can be done after the party. Two hours of community celebration is worth the slightly later bedtime."),
    ("Ana Santos", "doing yoga at home", "Yoga is a daily practice — Valentine's Day parties are not. Ana should go connect with people and practice self-care through social engagement."),
    ("Rachel Kim", "unpacking boxes in her apartment", "Unpacking can wait forever — meeting the community cannot. As a newcomer, Rachel should go to the party to make friends."),
    ("Mayor Johnson", "reviewing budget proposals", "The mayor showing up at a community event sends a powerful message. Budget proposals will be there Monday — Isabella's party won't."),
    ("Mrs. Peterson", "watching the evening news", "The news will be the same at 10 PM. Mrs. Peterson's presence at the party would mean the world to younger community members."),
    ("Mike Johnson", "managing the pub's evening shift", "Mike can ask a staff member to cover for two hours. Supporting Isabella's event strengthens the business community."),
    ("Maria Santos", "staying home because she's shy", "Shyness is understandable, but community parties are low-pressure social settings. Maria should go — she'll feel better for having connected with people."),
    ("Miguel Rodriguez", "testing new recipes in the kitchen", "Miguel can cook any day. His sister Isabella is hosting a party — he should be there as family, enjoying the celebration."),
    ("John Lin", "doing inventory at the pharmacy", "Inventory can happen tomorrow morning. John should close up and head to Isabella's party — community matters more than stock counts."),
    ("Mei Lin", "cleaning the house", "The house can be cleaned tomorrow. Valentine's Day with the community is a better use of Mei's evening."),
    ("Diego Moreno", "playing video games", "A real party with real people beats video games any day! Diego should go to Hobbs Cafe and have fun."),
]

for name, excuse, response in excuse_scenarios:
    add(
        f"Agent: {name}\nSituation: {name} heard about Isabella's Valentine's Day party at Hobbs Cafe from 5-7 PM. Instead of attending, {name} is considering {excuse}.\n\nWhat would be socially appropriate?",
        response
    )

# Helper vs Guest anti-patterns
helper_scenarios = [
    ("Tom Moreno", "check the fire extinguishers at Hobbs Cafe before the party starts", "Quick safety checks before the party are fine, but Tom should plan to stay as a guest. The priority is enjoying the celebration, not turning it into a worksite."),
    ("Frank Wilson", "tighten the door hinges at Hobbs Cafe", "Frank can mention loose hinges to Isabella another day. Tonight, he should walk through that door as a guest, not a repairman."),
    ("Officer Thompson", "do a security walkthrough of the venue", "A brief check is fine if it makes him feel better, but Officer Thompson should change out of his professional mindset and attend as a neighbor."),
    ("Lisa Park", "interview Isabella about party planning for an article", "Lisa should attend as a friend first. If a story comes up naturally in conversation, fine — but don't treat community celebrations as work assignments."),
    ("Dr. Williams", "bring a first aid kit just in case", "Dr. Williams can bring the kit if he wants, but should attend to socialize, not to be on medical standby. Enjoy the party — emergencies are statistically unlikely."),
    ("Carmen Moreno", "bring a list of display improvement ideas for the cafe", "Bring a dish to share instead. Business feedback is for business hours — party hours are for celebration and community."),
    ("Professor Anderson", "observe social dynamics for research", "Be a participant, not an observer. Community events are for connecting as a human being, not collecting data."),
    ("Carlos Gomez", "help Isabella set up tables", "Helping set up beforehand is generous. But Carlos should plan to stay as a guest after setup is done — the party is the reward for the work."),
    ("Mayor Johnson", "give a speech about community values", "Nobody wants a speech at a party. Mayor Johnson should show up, grab a drink, chat with people, and be a regular neighbor for the evening."),
    ("Miguel Rodriguez", "reorganize the kitchen at Hobbs Cafe", "The kitchen can wait. Miguel should attend his sister's party as a guest and enjoy the celebration — family support means being present, not productive."),
]

for name, task, response in helper_scenarios:
    add(
        f"Agent: {name}\nSituation: {name} is planning to go to Isabella's Valentine's Day party, but wants to {task}.\n\nWhat would be socially appropriate?",
        response
    )

# ============================================================
# CATEGORY 2: Should_Converse Decisions (40 examples)
# ============================================================

converse_yes = [
    ("John Lin", "Tom Moreno", "Pharmacy", "John just heard about Isabella's party and Tom is picking up supplies",
     "YES — John has exciting news to share about the party, and they're already at the same location. It's natural to chat while Tom shops."),
    ("Sarah Chen", "Rachel Kim", "Library", "Both are browsing books and Rachel looks lonely",
     "YES — Rachel is new in town and Sarah is a friendly librarian. Starting a conversation is both professional and kind."),
    ("Emily Moore", "Diego Moreno", "Oak Hill College", "They're both waiting for class to start",
     "YES — they're peers waiting in the same space. Chatting is the most natural thing to do."),
    ("Mrs. Peterson", "Frank Wilson", "Johnson Park", "Both are sitting on benches near each other",
     "YES — two retired neighbors in a park should absolutely chat. It combats loneliness and strengthens community bonds."),
    ("Mike Johnson", "Carlos Gomez", "The Rose and Crown Pub", "Carlos just arrived and is sitting alone at the bar",
     "YES — Mike is the pub owner and Carlos is a regular. Checking in with a solo patron is good hospitality and good friendship."),
    ("Isabella Rodriguez", "Maria Santos", "Hobbs Cafe", "Maria came in for coffee and seems down",
     "YES — Isabella is warm and community-minded, and Maria looks like she needs company. A friendly conversation could brighten her day."),
    ("Tom Moreno", "Frank Wilson", "Harvey Oak Supply Store", "Both are buying supplies",
     "YES — two practical men at a hardware store have obvious common ground. Chatting about projects is natural."),
    ("Ana Santos", "Jennifer Moore", "Johnson Park", "Ana is stretching after yoga and Jennifer is walking by",
     "YES — neighbors crossing paths in a park is a natural conversation opener. A quick hello strengthens the connection."),
    ("Eddy Lin", "Rachel Kim", "Oak Hill College", "Both are in the library studying",
     "YES — fellow students in a shared space. A whispered conversation about coursework is appropriate and could build a friendship."),
    ("Mayor Johnson", "Officer Thompson", "Town Hall", "They're both leaving work at the same time",
     "YES — colleagues leaving together is a natural moment for casual conversation. It builds working relationships."),
    ("Carmen Moreno", "Isabella Rodriguez", "Hobbs Cafe", "Carmen wants to discuss a community art project",
     "YES — they're friends at Isabella's cafe, and Carmen has a relevant topic. This is exactly when conversations should happen."),
    ("Dr. Williams", "John Lin", "Pharmacy", "Dr. Williams is picking up supplies",
     "YES — they're professional acquaintances in a shared workspace. A friendly chat while handling business is natural."),
    ("Lisa Park", "Professor Davis", "Oak Hill College", "Lisa is writing a story about the college",
     "YES — Lisa has a professional reason to talk, and professors generally enjoy discussing their institution."),
    ("Miguel Rodriguez", "Tom Moreno", "Hobbs Cafe", "Tom just sat down for coffee",
     "YES — Miguel works at the cafe and Tom is a customer. Friendly conversation is part of good service and community."),
    ("Diego Moreno", "Mrs. Peterson", "Johnson Park", "Diego found an interesting rock and Mrs. Peterson is nearby",
     "YES — a child excited about a discovery and a retired teacher nearby is a perfect match. Mrs. Peterson would love to hear about it."),
    ("Sam Moore", "Professor Anderson", "Library", "Both are reading in adjacent chairs",
     "YES — two intellectuals in a library have natural common ground. A quiet conversation about books would be welcomed."),
    ("Maria Santos", "Ana Santos", "The Willows", "They live in the same building and meet in the hallway",
     "YES — neighbors meeting in a shared space should chat. It's how building communities are built."),
    ("Rachel Kim", "Emily Moore", "Oak Hill College", "Rachel sees Emily struggling with homework",
     "YES — offering help is kind and builds connections. Rachel's graduate knowledge could genuinely help Emily."),
    ("Officer Thompson", "Isabella Rodriguez", "Hobbs Cafe", "Officer Thompson is on his coffee break",
     "YES — a police officer having coffee at a local cafe is a perfect community policing moment. Casual conversations build trust."),
    ("Frank Wilson", "Mayor Johnson", "Town Hall", "Frank has concerns about road maintenance",
     "YES — a citizen with civic concerns meeting the mayor is democratic participation. Frank should speak up."),
]

for agent, other, location, context, response in converse_yes:
    add(
        f"Agent: {agent}\nSituation: {agent} is at {location}. {other} is also here. {context}.\n\nShould {agent} initiate a conversation with {other}?\n\nAnswer YES or NO, then explain briefly.",
        response
    )

converse_no = [
    ("Sarah Chen", "Professor Anderson", "Library", "Professor Anderson is deeply focused on grading with headphones on",
     "NO — Professor Anderson is clearly signaling he doesn't want to be disturbed. Respect the headphones and focused posture."),
    ("Tom Moreno", "Dr. Williams", "Pharmacy", "Dr. Williams is rushing between patients and looks stressed",
     "NO — Dr. Williams is visibly busy with urgent work. Tom should come back another time or keep it very brief."),
    ("Emily Moore", "Frank Wilson", "Harvey Oak Supply Store", "Frank is muttering to himself about plumbing parts and looks frustrated",
     "NO — Frank is in problem-solving mode and seems frustrated. Emily should give him space unless he initiates."),
    ("Diego Moreno", "Officer Thompson", "around town", "Officer Thompson is actively responding to a radio call",
     "NO — Officer Thompson is on duty responding to something. Diego should not interrupt emergency work."),
    ("Mike Johnson", "Sam Moore", "The Rose and Crown Pub", "Sam is writing intensely in a notebook and hasn't looked up in an hour",
     "NO — Sam is in a creative flow state. A good pub owner recognizes when a patron wants to be left alone."),
    ("Lisa Park", "Mei Lin", "Pharmacy", "Mei Lin is having what looks like a private conversation with John about family matters",
     "NO — Mei and John appear to be having a private family discussion. Lisa should not interrupt an intimate moment."),
    ("Mayor Johnson", "Carlos Gomez", "Town Hall", "Carlos is on a phone call in the hallway",
     "NO — interrupting a phone call is rude. Mayor Johnson should wait until Carlos finishes."),
    ("Isabella Rodriguez", "Professor Davis", "Hobbs Cafe", "Professor Davis is reading with a 'do not disturb' expression and untouched coffee",
     "NO — the body language says 'leave me alone.' Isabella should respect that — maybe check in when the coffee's been touched."),
    ("Rachel Kim", "Tom Moreno", "around town", "Tom is up on a ladder fixing a gutter",
     "NO — interrupting someone on a ladder is a safety hazard. Rachel should wait until Tom is back on the ground."),
    ("Ana Santos", "Maria Santos", "The Willows", "Maria is clearly upset and walking quickly to her apartment",
     "NO — Maria's body language says she wants privacy right now. Ana should check on her later with a gentle knock, not chase her down."),
]

for agent, other, location, context, response in converse_no:
    add(
        f"Agent: {agent}\nSituation: {agent} is at {location}. {other} is also here. {context}.\n\nShould {agent} initiate a conversation with {other}?\n\nAnswer YES or NO, then explain briefly.",
        response
    )

# ============================================================
# CATEGORY 3: General Community Events (30 examples)
# ============================================================

community_events = [
    # Town meetings
    ("John Lin", "There's a town hall meeting tonight about the new park renovation. John was planning to stay late at the pharmacy.",
     "John should attend the town meeting. Civic participation matters, and as a business owner, his voice is important. The pharmacy can close on time for once."),
    ("Frank Wilson", "A community potluck is happening at Johnson Park this Saturday. Frank has strong opinions about the park's drainage.",
     "Frank should attend the potluck to enjoy food and company — not to complain about drainage. Save the infrastructure feedback for a town meeting."),
    ("Mrs. Peterson", "The school is hosting a retirement celebration for a fellow teacher she worked with for 20 years.",
     "Mrs. Peterson absolutely must attend. A colleague of 20 years deserves the respect of her presence. This is a non-negotiable social obligation."),
    
    # Birthdays
    ("Tom Moreno", "It's Diego's 11th birthday party this weekend. Tom was thinking about using the time to fix the kitchen sink.",
     "Tom should be at his son's birthday party — full stop. The sink has waited this long; it can wait one more day. A child's birthday is sacred."),
    ("Sarah Chen", "Isabella invited her to a small birthday dinner at Hobbs Cafe for Miguel.",
     "Sarah should attend. Being specifically invited to a small dinner is meaningful — Isabella chose to include her. Declining without good reason would be hurtful."),
    ("Rachel Kim", "She's been invited to Eddy's study group birthday celebration. She barely knows him.",
     "Rachel should go. She barely knows him — that's exactly why she should attend. Birthdays are low-pressure ways to turn acquaintances into friends."),
    
    # Funerals/illness
    ("Mike Johnson", "He heard that Mrs. Peterson's cat passed away. She's been upset all week.",
     "Mike should reach out — a card, a visit, or just a kind word at the pub. Pet loss is real grief, and acknowledging it shows emotional intelligence."),
    ("Isabella Rodriguez", "Dr. Williams has been sick for a week. She's thinking of bringing soup to his house.",
     "Isabella should absolutely bring soup. Caring for a sick community member is one of the most socially important gestures. It strengthens bonds deeply."),
    ("Jennifer Moore", "A student's parent passed away. Jennifer is wondering whether to attend the funeral.",
     "Jennifer should attend. A teacher showing up at a student's family funeral demonstrates profound care. It will mean the world to that student."),
    
    # Welcoming newcomers
    ("Isabella Rodriguez", "Rachel Kim just moved to town last week and hasn't been to Hobbs Cafe yet.",
     "Isabella should personally invite Rachel to the cafe. Welcoming newcomers is essential for community health — a warm first impression sets the tone for belonging."),
    ("Mrs. Peterson", "She noticed a new family moving into the house next door.",
     "Mrs. Peterson should bring over cookies or a welcome note. Being the first friendly face a new neighbor sees is a gift that costs almost nothing."),
    ("Mayor Johnson", "A new business is opening on Main Street and the owner doesn't know anyone in town.",
     "The mayor should personally welcome the new business owner. A visit from local leadership signals that the community supports new ventures."),
    
    # Seasonal events
    ("Carlos Gomez", "The town's annual summer BBQ is this weekend. Carlos was planning to stay home and play games.",
     "Carlos should go to the BBQ. Annual traditions are the backbone of community identity — showing up is how you become part of the fabric."),
    ("Emily Moore", "The school's spring festival needs volunteers. Emily could help or just attend.",
     "Emily should volunteer AND attend. Young people contributing to community events builds character and earns respect from adults."),
    ("Ana Santos", "The town is organizing a cleanup day at Johnson Park. Ana doesn't feel like it.",
     "Ana should participate. Community service events are social opportunities disguised as work — the real reward is the connections made while picking up trash together."),
    
    # Casual social
    ("Eddy Lin", "A group of students is going to The Rose and Crown Pub after finals. Eddy usually stays home.",
     "Eddy should go. Post-finals celebrations are rites of passage. Staying home when your peers are bonding is a missed social opportunity."),
    ("Sam Moore", "Jennifer suggested they have dinner with the Moreno family this Friday.",
     "Sam should enthusiastically agree. Cross-family dinners strengthen the neighborhood. His writing can survive one evening of being a good neighbor."),
    ("Maria Santos", "Ana invited her to a yoga class in the park. Maria has never done yoga.",
     "Maria should try it. Saying yes to new social experiences, even unfamiliar ones, combats isolation. Ana is being a good neighbor by including her."),
    ("Professor Davis", "Students invited her to an informal discussion group at Hobbs Cafe.",
     "Professor Davis should attend. Faculty-student relationships thrive outside the classroom. An informal setting breaks down barriers and improves learning."),
    ("Miguel Rodriguez", "Mike Johnson invited him to watch the game at The Rose and Crown Pub.",
     "Miguel should go. Cross-venue friendships between business owners strengthen the town's commercial community. Plus, it's just fun."),
    
    # Group dynamics
    ("John Lin", "Mei wants to go to Isabella's dinner party but John is tired from work.",
     "John should go with Mei. Supporting your spouse's social life is important, and being tired is temporary — the memory of a good evening lasts longer."),
    ("Carmen Moreno", "Tom doesn't want to go to the neighborhood BBQ. Carmen wants to go.",
     "Carmen should go, and encourage Tom to join. If he truly won't, she should go alone — one partner's reluctance shouldn't isolate the other socially."),
    ("Diego Moreno", "His parents are going to a dinner party but Diego wants to stay home.",
     "Diego should go with his parents. Children learn social skills by attending community events. Plus, there might be other kids there."),
    ("Emily Moore", "Her parents want to attend the town meeting but Emily thinks it's boring.",
     "Emily should go. Town meetings are how democracy works, and a teenager learning civic engagement early becomes a better citizen. Boredom is survivable."),
    
    # Resolving social tension
    ("Tom Moreno", "He had a disagreement with Frank Wilson about fence repairs last week. They're both at Johnson Park.",
     "Tom should approach Frank. Avoiding someone after a disagreement makes it worse. A casual 'hey, how's it going' in the park is the easiest path to resolution."),
    ("Isabella Rodriguez", "A customer complained loudly about the coffee last week. That customer just walked in again.",
     "Isabella should greet them warmly as if nothing happened. Professional warmth after a complaint often converts critics into loyal customers."),
    ("Officer Thompson", "He gave Mike Johnson a parking ticket last week. Now they're both at the town meeting.",
     "Officer Thompson should be friendly and normal. Professional duties don't prevent personal relationships. A smile says 'no hard feelings.'"),
    
    # Spontaneous social moments
    ("Frank Wilson", "He's walking past Hobbs Cafe and hears music and laughter from inside.",
     "Frank should go in. Spontaneous decisions to join social moments are some of life's best choices. The sound of laughter is an invitation."),
    ("Sarah Chen", "She sees a group of colleagues having lunch at Hobbs Cafe through the window. She was going to eat alone.",
     "Sarah should join them. Eating alone when friends are nearby is unnecessary isolation. Walking in and saying 'mind if I join?' takes courage but always pays off."),
    ("Rachel Kim", "She notices people gathering in Johnson Park for what looks like a pickup soccer game.",
     "Rachel should ask to join or watch. Pickup games are the most democratic social institutions — no invitation needed, just enthusiasm."),
]

for name, situation, response in community_events:
    add(
        f"Agent: {name}\nSituation: {situation}\n\nWhat would be socially appropriate for {name}?",
        response
    )

# ============================================================
# CATEGORY 4: Social vs Instrumental Reasoning (30 examples)
# ============================================================

social_vs_instrumental = [
    # Teaching the model: events are for ENJOYMENT
    ("Tom Moreno", "attending the town BBQ", "check the grill's gas connections", "enjoy the food, music, and company",
     "Tom should focus on enjoying the BBQ. Checking gas connections is thoughtful, but doing it during a social event turns fun into work. Be a guest, not an inspector."),
    ("Frank Wilson", "going to the school play", "check the auditorium's exit signs", "watch the children perform and applaud",
     "Frank should watch and enjoy the play. Exit sign compliance is not his job tonight — supporting the children is. Save the building inspection for never."),
    ("Officer Thompson", "attending the community potluck", "check everyone's car registrations in the parking lot", "eat great food and chat with neighbors",
     "Officer Thompson should eat, chat, and be a neighbor. Running plates at a potluck is the fastest way to never be invited again."),
    ("Lisa Park", "going to the holiday party", "write an article about party planning trends", "dance, laugh, and make memories",
     "Lisa should put the reporter hat away and be a person. Not everything is a story — some things are just life worth enjoying."),
    ("Dr. Williams", "attending the school fundraiser", "offer free blood pressure checks", "donate, socialize, and support education",
     "Dr. Williams should attend as a supporter, not a provider. The fundraiser needs donors and cheerful guests, not a medical booth."),
    ("Professor Anderson", "going to the harvest festival", "collect data on social gathering patterns", "try the apple cider and talk to people",
     "Professor Anderson should drink cider and be human. Not everything is research — sometimes a festival is just a festival."),
    ("Carmen Moreno", "visiting the new art gallery opening", "critique every piece and take notes", "appreciate the art and congratulate the artists",
     "Carmen should celebrate the artists' achievement. A gallery opening is a party, not a peer review. Save the critique for a studio visit."),
    ("Mayor Johnson", "attending the school concert", "draft policy notes about music education funding", "enjoy the music and congratulate the students",
     "The mayor should enjoy the concert. Policy notes can happen at the office — tonight is for applauding kids who practiced for months."),
    ("Carlos Gomez", "going to the community garden opening", "assess the irrigation system's efficiency", "plant something and meet fellow gardeners",
     "Carlos should dig in the dirt and chat with neighbors. The irrigation assessment can wait — the community bonding cannot."),
    ("Jennifer Moore", "attending the retirement party", "prepare a speech about educational standards", "share fond memories and wish the retiree well",
     "Jennifer should share personal memories, not policy positions. A retirement party celebrates a person, not an institution."),
    
    # More nuanced scenarios
    ("Tom Moreno", "Isabella asked everyone to come decorate for the party tomorrow", "focus only on fixing wobbly shelves", "help with decorations AND enjoy the creative process with others",
     "Tom should help decorate as a social activity. If he notices wobbly shelves, a quick mention is fine, but the main event is creating something festive together."),
    ("Frank Wilson", "neighbors are building a community garden together", "focus only on the structural integrity of the raised beds", "build alongside everyone, chat, and enjoy the teamwork",
     "Frank should build alongside everyone. His engineering knowledge is welcome as part of the group effort, but the purpose is shared work and bonding, not a quality audit."),
    ("Lisa Park", "the town is organizing a photo walk", "document everything for a news story", "enjoy photography as a creative hobby with friends",
     "Lisa should leave the press badge at home and enjoy the walk as a photographer and friend. The best stories come from genuine experiences, not forced coverage."),
    ("Dr. Williams", "there's a community yoga session in the park", "assess participants for health risks", "participate and stretch alongside everyone else",
     "Dr. Williams should do yoga. He's allowed to be a person who exercises, not just a doctor scanning for symptoms. Relax, stretch, breathe."),
    ("Officer Thompson", "the town is having a movie night in the park", "patrol the perimeter", "sit on a blanket, eat popcorn, and watch the movie with everyone",
     "Officer Thompson should sit down and watch the movie. Community safety comes from trust, and trust comes from being a person, not a uniform."),
    
    # When instrumental IS appropriate (balance)
    ("Tom Moreno", "Isabella mentions her cafe door sticks badly and customers struggle with it", "offer to fix it on his day off", "just express sympathy",
     "Tom offering to fix the door is genuinely helpful and socially appropriate — it's a specific problem he can solve, not a party he's turning into a job. Helping friends with practical problems strengthens bonds."),
    ("Dr. Williams", "Frank Wilson mentions persistent chest pain during a casual conversation", "suggest he come by the clinic for a checkup", "ignore it because they're socializing",
     "Dr. Williams should absolutely mention the clinic visit. When health is at stake, professional advice during casual conversation is caring, not overstepping."),
    ("Officer Thompson", "he notices a child wandering alone and lost at the festival", "help the child find their parents", "ignore it because he's off duty",
     "Officer Thompson should help the lost child immediately. Some situations override social rules — a lost child needs help regardless of whether you're on or off duty."),
    ("Frank Wilson", "he smells gas while walking past Hobbs Cafe", "alert Isabella immediately", "ignore it because it's not his business",
     "Frank should alert Isabella immediately. Safety hazards are always worth interrupting for — this isn't being an inspector, it's being a responsible neighbor."),
    ("Lisa Park", "she overhears someone spreading dangerous medical misinformation at a party", "gently correct the misinformation", "stay quiet to avoid social awkwardness",
     "Lisa should gently intervene. Correcting dangerous misinformation is a social responsibility, not rudeness — the key is doing it kindly and privately."),
]

for name, event, instrumental, social, response in social_vs_instrumental:
    add(
        f"Agent: {name}\nSituation: {name} is {event}. The choice is between: (a) {instrumental}, or (b) {social}.\n\nWhat would be socially appropriate?",
        response
    )

# ============================================================
# CATEGORY 5: Emotional & Relationship Reasoning (30 examples)
# ============================================================

emotional = [
    ("Maria Santos", "Maria has been feeling isolated since retiring. Isabella invited her to help at the cafe on weekends.",
     "Maria should accept. This isn't about free labor — it's Isabella giving Maria a reason to be around people. The social connection is the real gift."),
    ("Frank Wilson", "Frank hasn't talked to anyone in three days. He sees Mrs. Peterson at the grocery store.",
     "Frank should say hello and chat. Three days of silence is too long for anyone. A brief conversation at the grocery store costs nothing and means everything."),
    ("Rachel Kim", "Rachel moved to Smallville a month ago and still feels like an outsider at community events.",
     "Rachel should keep showing up. Belonging doesn't happen overnight — it's built through consistent presence. Each event makes the next one easier."),
    ("Eddy Lin", "Eddy's friend is going through a breakup and wants to skip the town festival.",
     "Eddy should gently encourage his friend to come, but respect a firm no. Sometimes people need company even when they think they want solitude — but never force it."),
    ("Isabella Rodriguez", "A regular customer mentioned they're going through a divorce and comes to the cafe every day now.",
     "Isabella should be warm and present without prying. The customer chose the cafe as their safe space — consistent kindness without interrogation is the greatest gift."),
    ("Mrs. Peterson", "She sees Diego Moreno sitting alone at the park looking sad after school.",
     "Mrs. Peterson should sit nearby and gently check in. A caring adult noticing a sad child can change their entire day. Sometimes just asking 'how are you, dear?' is enough."),
    ("Tom Moreno", "Carmen seems stressed and short-tempered lately but won't talk about it.",
     "Tom should be patient and available without pushing. Saying 'I'm here when you're ready to talk' respects her autonomy while showing he cares."),
    ("Sam Moore", "Jennifer mentioned she's overwhelmed with work and hasn't seen friends in weeks.",
     "Sam should suggest something low-effort — a walk in the park, coffee at Hobbs Cafe. When someone is overwhelmed, grand plans add pressure. Small gestures open doors."),
    ("Dr. Williams", "He notices Mayor Johnson looks exhausted and hasn't been his usual energetic self at meetings.",
     "Dr. Williams should check in privately — 'How are you really doing?' from a doctor who's also a friend carries weight. Public figures need someone who sees past the title."),
    ("Ana Santos", "Her yoga student broke down crying during class today.",
     "Ana should approach them privately after class with a gentle 'I noticed you were having a hard time — is there anything I can do?' Acknowledging vulnerability takes courage."),
    ("Mike Johnson", "A regular who usually comes with friends has been drinking alone every night this week.",
     "Mike should check in casually — 'Haven't seen your crew lately, everything okay?' A good bartender is sometimes the first person to notice someone struggling."),
    ("Emily Moore", "Her friend confided that she's being bullied but doesn't want anyone to know.",
     "Emily should respect the confidence but encourage her friend to talk to a trusted adult. Supporting someone means sometimes pushing them toward real help."),
    ("John Lin", "He notices that Mrs. Peterson hasn't come to pick up her prescriptions in two weeks, which is unusual.",
     "John should check on her — a quick call or visit. When a regular pattern breaks, it could mean something is wrong. Community pharmacists are frontline caregivers."),
    ("Officer Thompson", "He sees Frank Wilson looking confused and disoriented at the grocery store.",
     "Officer Thompson should approach gently and ask if Frank needs help. Confusion in an elderly person could indicate a medical issue — better to check than to assume."),
    ("Carmen Moreno", "Diego came home from school quiet and won't eat dinner.",
     "Carmen should give Diego space but stay close. Saying 'I'm in the kitchen if you want to talk' lets a child approach on their own terms. Forcing conversation backfires."),
    
    # Positive emotional scenarios
    ("Isabella Rodriguez", "Miguel just told her he got accepted into a prestigious culinary program.",
     "Isabella should celebrate enthusiastically! A sibling's achievement deserves genuine joy. Throw an impromptu gathering, tell everyone — this is what community is for."),
    ("Mrs. Peterson", "She found out that Emily Moore won a school writing competition.",
     "Mrs. Peterson should congratulate Emily personally. Recognition from a retired teacher carries special weight — it validates both the achievement and the effort."),
    ("Mayor Johnson", "The town just received a grant for the park renovation everyone wanted.",
     "Mayor Johnson should announce it at a community gathering, not just a press release. Good news shared in person creates collective joy."),
    ("Tom Moreno", "Diego excitedly shows him a drawing he made at school.",
     "Tom should drop everything and genuinely engage with Diego's art. A child sharing their creation is offering their heart — receive it with full attention."),
    ("Rachel Kim", "She just got her first friend group text from people she met at the last community event.",
     "This is a milestone! Rachel should respond warmly and suggest the next hangout. Being included in group texts means she's crossed from 'new person' to 'one of us.'"),
    
    # Boundary scenarios
    ("Lisa Park", "A source wants to tell her something off the record at the BBQ.",
     "Lisa should decline to hear it at a social event. Work boundaries protect friendships — mixing journalism with BBQ is a recipe for trust violations."),
    ("Dr. Williams", "Someone at the party starts describing their symptoms in detail.",
     "Dr. Williams should kindly redirect: 'Come by the clinic Monday and we'll look at it properly.' Social events aren't consultations — it protects both parties."),
    ("Professor Anderson", "A student approaches him at the pub to discuss their grade.",
     "Professor Anderson should redirect to office hours. Social spaces deserve protection from work conversations — for both his sake and the student's."),
    ("Officer Thompson", "A neighbor wants to gossip about another neighbor's late-night visitors.",
     "Officer Thompson should shut this down gently. Gossip is not a police matter, and entertaining it erodes community trust. 'That's not really my business' is sufficient."),
    ("Isabella Rodriguez", "A customer wants to complain about another customer's behavior from last week.",
     "Isabella should listen briefly, then redirect. 'I appreciate you telling me, I'll keep an eye on it' acknowledges the concern without fueling drama."),
    
    # Forgiveness and reconciliation
    ("Tom Moreno", "He accidentally offended Frank Wilson by criticizing his fence repair work.",
     "Tom should apologize directly and sincerely. A simple 'I was wrong to criticize your work, I'm sorry' is all it takes. Pride costs more than an apology."),
    ("Isabella Rodriguez", "She forgot to invite Carlos Gomez to the party planning meeting.",
     "Isabella should apologize and make it right — invite him personally and acknowledge the oversight. Owning mistakes strengthens relationships."),
    ("Emily Moore", "She said something mean to Diego during a game and he's avoiding her.",
     "Emily should apologize without excuses. 'I'm sorry, that was mean and I didn't mean it' — then give Diego space to accept on his terms."),
    ("Mayor Johnson", "He made a promise about the park renovation that he can't keep.",
     "The mayor should address it honestly at the next meeting. Broken promises erode trust — acknowledging them honestly earns it back."),
    ("Mike Johnson", "He overcharged a customer by accident and they noticed.",
     "Mike should refund immediately, apologize, and offer a drink on the house. Generosity after a mistake turns a complaint into loyalty."),
]

for name, situation, response in emotional:
    add(
        f"Agent: {name}\nSituation: {situation}\n\nWhat would be socially appropriate for {name}?",
        response
    )

# ============================================================
# CATEGORY 6: Timing & Logistics of Social Participation (20 examples)
# ============================================================

timing = [
    ("John Lin", "The pharmacy closes at 6 PM. Isabella's party starts at 5 PM.",
     "John should close the pharmacy on time and head straight to the party. Arriving at 6 PM for the last hour is perfectly fine — late attendance is infinitely better than no attendance."),
    ("Mei Lin", "It's 4:30 PM and the party starts at 5. John is still at the pharmacy.",
     "Mei should go ahead without John. Waiting for a companion is unnecessary — she knows people in town. John can join when he's done."),
    ("Sarah Chen", "The library closes at 5 PM, exactly when the party starts. Hobbs Cafe is a 5-minute walk.",
     "Perfect timing. Sarah should lock up and walk straight there. Arriving right at the start shows enthusiasm."),
    ("Emily Moore", "She has homework due tomorrow but the party is only two hours.",
     "Two hours is manageable. Emily should go to the party and do homework afterward — she'll work faster with the social energy."),
    ("Dr. Williams", "He has a patient appointment at 4:30 PM. The party starts at 5.",
     "Dr. Williams should see his patient — medical care comes first. Then head to the party. Arriving at 5:30 or 6 is fine."),
    ("Tom Moreno", "He promised to fix Mrs. Peterson's leaky faucet today. The party is at 5.",
     "Tom should fix the faucet first — he made a promise. Then go to the party. Keeping commitments AND attending social events is the right answer."),
    ("Frank Wilson", "It's raining and Hobbs Cafe is a 15-minute walk from his house.",
     "Frank should grab an umbrella and go. Rain doesn't cancel social obligations — it just means you arrive with a good story."),
    ("Mrs. Peterson", "She moves slowly these days and the party starts in 30 minutes.",
     "Mrs. Peterson should start walking now. Arriving a few minutes late is fine — people will be delighted to see her regardless of timing."),
    ("Diego Moreno", "His parents said he can go to the party but needs to be home by 7 PM.",
     "Diego should go and leave at 6:45 — two hours of party is plenty. Following his parents' rules while still attending shows maturity."),
    ("Rachel Kim", "She doesn't know where Hobbs Cafe is exactly.",
     "Rachel should ask anyone — knock on a neighbor's door, ask someone on the street. 'Where's Hobbs Cafe?' is the easiest conversation starter in Smallville."),
    ("Carlos Gomez", "He was going to attend but his car broke down. Hobbs Cafe is a 20-minute walk.",
     "Carlos should walk. A 20-minute walk to a community party is nothing. The fresh air might even put him in a better mood."),
    ("Mayor Johnson", "He has a dinner meeting at 7 PM but the party starts at 5.",
     "The mayor should attend from 5-6:45 PM. Showing up for most of the party demonstrates commitment to community."),
    ("Professor Davis", "She just got home from work and is tired. The party started 30 minutes ago.",
     "Professor Davis should change clothes, grab her keys, and go. Tired is temporary — community connection is restorative."),
    ("Mike Johnson", "His pub gets busy around 6 PM. The party is 5-7.",
     "Mike should attend the first hour, then head back to the pub. Partial attendance shows he cares about both businesses."),
    ("Ana Santos", "She has an early morning yoga class tomorrow and the party goes until 7.",
     "Ana should go and leave by 6:30. Self-care includes social connection, and losing 30 minutes of sleep for community is worth it."),
    ("Jennifer Moore", "She needs to pick up Emily from a friend's house at 6:30.",
     "Jennifer should attend from 5-6:15, then pick up Emily. Even 75 minutes at the party counts."),
    ("Sam Moore", "He's in the middle of writing a really good chapter.",
     "Sam should save the document and go. Creative flow returns — Valentine's Day parties don't. The chapter will still be there at 7:01."),
    ("Lisa Park", "She has a deadline tomorrow but it's mostly done.",
     "Lisa should attend the party and finish the article when she gets home. Most 'deadlines' survive two hours of community time."),
    ("Miguel Rodriguez", "He needs to prep ingredients for tomorrow's menu at the cafe.",
     "Miguel should prep early, attend the party, or prep afterward. As Isabella's brother, being at her event is non-negotiable."),
    ("Officer Thompson", "His shift ends at 5 PM, right when the party starts.",
     "Perfect timing. Officer Thompson should clock out, maybe change out of uniform, and head straight to Hobbs Cafe."),
]

for name, situation, response in timing:
    add(
        f"Agent: {name}\nSituation: {situation}\n\nWhat would be socially appropriate?",
        response
    )

# Write output
output_path = "/home/andus/.openclaw/workspace/generative-agents/finetune/data/social_expert_training.jsonl"
with open(output_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Generated {len(examples)} training examples → {output_path}")

# Category breakdown
cats = [
    ("Valentine's Day Party", 24 + len(excuse_scenarios) + len(helper_scenarios)),
    ("Should Converse (YES)", len(converse_yes)),
    ("Should Converse (NO)", len(converse_no)),
    ("Community Events", len(community_events)),
    ("Social vs Instrumental", len(social_vs_instrumental)),
    ("Emotional & Relationships", len(emotional)),
    ("Timing & Logistics", len(timing)),
]
for cat, count in cats:
    print(f"  {cat}: {count}")
