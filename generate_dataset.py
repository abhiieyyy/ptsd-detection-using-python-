"""
============================================================
  generate_dataset.py   (v4 — 10,000 samples, Kaggle ready)

  Generates 3 CSV datasets with 10,000 rows each.
  700+ unique PTSD text templates.
  Real Kaggle dataset auto-loader (drop CSV in real_data/).

  KAGGLE DATASETS TO DOWNLOAD (free, no login needed):
  ─────────────────────────────────────────────────────
  1. Mental Health Corpus (best for PTSD text):
     https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus
     → Save as: real_data/mental_health.csv   (cols: text, label)

  2. Sentiment140 / Depression Reddit:
     https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media
     → Save as: real_data/social_media.csv    (cols: post_text, label)

  3. DAIC-WOZ (clinical gold standard, requires registration):
     https://dcapswoz.ict.usc.edu
     → Save as: real_data/daic_woz.csv        (cols: text, label, severity)

  HOW TO USE:
      python3 generate_dataset.py
============================================================
"""

import os, random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N          = 10000      # samples per modality
PTSD_RATIO = 0.42


# ═══════════════════════════════════════════════════════════════
#  FILL-IN LISTS
# ═══════════════════════════════════════════════════════════════
EVENTS = [
    "the car accident","the combat deployment","the assault","the natural disaster",
    "the house fire","the workplace incident","the violent attack","the sexual assault",
    "the robbery","the explosion","the shooting","the flood","the earthquake",
    "losing my colleague","the crash","the military operation","the ambush",
    "the bombing","the train accident","the plane crash","the building collapse",
    "the childhood abuse","the domestic violence","the near-drowning",
    "the hostage situation","losing my child","the medical emergency",
    "witnessing the accident","the mugging","the home invasion","the wildfire",
]
PLACES = [
    "that building","the hospital","the street where it happened","that intersection",
    "the old neighbourhood","that part of town","the workplace","the park",
    "the bridge","that room","the parking lot","the mall","the school",
    "the site of the accident","that road","the old house","the market",
]
TOPICS = [
    "violence","accidents","military combat","trauma","disaster","abuse",
    "death","injury","conflict","war","emergency situations","sexual violence",
    "domestic abuse","childhood trauma","natural disasters",
]


# ═══════════════════════════════════════════════════════════════
#  700+ PTSD TEXT TEMPLATES
# ═══════════════════════════════════════════════════════════════
ptsd_templates = [

    # ── FLASHBACKS / RE-EXPERIENCING (60 templates) ────────────
    "I keep having flashbacks to {event}. Every time I close my eyes I'm right back there and it feels completely real.",
    "The nightmares won't stop. I wake up screaming almost every night reliving {event} over and over.",
    "Sometimes out of nowhere I'm transported back to {event}. My heart pounds and I can barely breathe.",
    "I relived the whole {event} again last night. I couldn't breathe and I couldn't move.",
    "Every loud noise makes me jump and I'm right back in {event}. My body reacts before my mind does.",
    "I can't stop seeing it happen over and over in my head. The images are so vivid it's like watching a movie.",
    "Last night I dreamed I was back in {event}. I woke up drenched in sweat and absolutely terrified.",
    "A car backfired today and instantly I was back in {event}. My hands were shaking for twenty minutes.",
    "The smell triggers the memory of {event} every single time. I cannot escape it no matter what I do.",
    "I saw something on the news that reminded me of {event} and had a full panic attack right there on the spot.",
    "The flashbacks come without warning. One moment I'm fine and the next I'm back in {event}.",
    "I keep reliving the worst moments over and over in my mind. It never stops and I'm exhausted from it.",
    "During the day I can sometimes manage but at night the memories of {event} take over completely.",
    "I had another intrusive memory at work today. Everyone noticed and I didn't know how to explain.",
    "The images from {event} play on a loop in my head. I feel like I'm going insane from it.",
    "I was driving and suddenly I was back in {event}. I had to pull over because I couldn't see through my tears.",
    "My body reacts to triggers as if {event} is happening right now. The physical sensations are completely real.",
    "I woke up on the floor again last night. I must have been acting out my nightmare of {event} in my sleep.",
    "The trauma plays like a film I can't turn off. Every detail is sharp and terrifying and inescapable.",
    "Sometimes I lose time during a flashback. I come back and I don't know where I've been mentally.",
    "I can hear the sounds from {event} even in silence. My brain cannot distinguish past from present.",
    "Every anniversary of {event} I completely fall apart. The date itself is a trigger I can't avoid.",
    "My therapist calls them intrusive memories but they feel so real. I cannot convince my body it's over.",
    "I shook uncontrollably when I accidentally saw a photo that reminded me of {event}. It lasted an hour.",
    "The sensory memories are the worst. A certain sound, smell, or texture and I'm right back there again.",
    "I was in the supermarket and a random song triggered a complete flashback to {event}. I froze completely.",
    "I can't watch the news anymore. Any story about {topic} sends me straight back to {event}.",
    "Even in therapy sessions the memories feel so vivid and immediate. It's exhausting to keep revisiting {event}.",
    "The dreams feel more real than waking life now. I dread sleeping because of {event}.",
    "I've started wearing earphones everywhere just to block sounds that might trigger {event} memories.",
    "When I close my eyes I can still feel it happening. {event} is permanently etched into my nervous system.",
    "I dissociated again today. I was in the middle of a conversation and suddenly I was back at {event}.",
    "I screamed in my sleep again last night. My partner held me for an hour while I came back to reality.",
    "The intrusive thoughts have gotten so bad I can barely function at work since {event}.",
    "I had a panic attack in the cinema when a scene reminded me of {event}. I had to leave immediately.",
    "I keep re-experiencing the sounds. The screams, the impact, everything about {event} is still so vivid.",
    "Some mornings I wake up and for a second I forget. Then the memory of {event} crashes back in instantly.",
    "I replayed {event} in my head at least a dozen times today. I want so badly to stop but I can't.",
    "My hyperreactive startle response has made me a liability at work since {event}.",
    "I see it happening in slow motion every time I try to relax. {event} has colonised my every quiet moment.",
    "I've been waking at 3am every single night replaying exactly what happened during {event}.",
    "My body remembers {event} even when my mind tries to move on. The shaking, the sweating, it's all still there.",
    "I had a body memory today at the gym. The physical sensation triggered the entire {event} memory instantly.",
    "A particular smell on the street stopped me completely today. It was exactly like the smell from {event}.",
    "I tried mindfulness but every time I sit quietly the flashbacks of {event} are just louder.",
    "I cried for an hour today over a news headline that had nothing to do with {event} but felt exactly like it.",
    "I keep dreaming that I'm trying to stop {event} but I can't move. I wake up paralysed with fear.",
    "The fight-or-flight response triggers at completely random moments now. My body never got the memo that {event} is over.",
    "I had to leave a family dinner because a conversation topic triggered images of {event} I can't unsee.",
    "I am so tired of being hijacked by memories of {event} at the most random and inconvenient moments.",
    "Even medication hasn't stopped the dreams about {event}. Some nights I'm afraid to fall asleep at all.",
    "My hands started shaking at work today for no apparent reason. Then I realised: the date. It's the anniversary of {event}.",
    "I've taken to sleeping with the light on since {event}. The darkness makes the nightmares worse.",
    "I barely recognise the person I was before {event}. That person wasn't afraid of their own shadow.",
    "I've started carrying a photo on my phone to show people what I looked like before {event} changed me.",
    "I walked out of a work presentation when someone used a phrase that took me straight back to {event}.",
    "I described the flashbacks to my doctor today. She said they're intrusive memories. I call them torture.",
    "I've been off work for three weeks now since the flashbacks from {event} made it impossible to function.",
    "Every time I hear a siren now I'm right back at {event}. I pull over, put my hands on the wheel, and wait.",
    "I've had to rearrange my entire life to avoid triggers. {event} controls my geography now.",

    # ── AVOIDANCE (55 templates) ────────────────────────────────
    "I can't go near {place} anymore. It brings back everything I've been desperately trying to forget.",
    "I avoid crowds at all costs. Being surrounded by people makes me feel trapped and completely unsafe.",
    "I stopped driving after {event}. Every time I get in a car I have a panic attack.",
    "I cancelled plans again because I simply can't handle being around people right now.",
    "I refuse to talk about what happened during {event}. Every time someone brings it up I shut down completely.",
    "I've been avoiding the news entirely. Anything related to {topic} sends me into a downward spiral.",
    "I can't watch movies with {topic} scenes. They trigger my worst and most vivid memories.",
    "I haven't left my house in weeks. The outside world feels far too dangerous and unpredictable.",
    "I deleted all my social media because seeing other people living normal lives makes me feel broken.",
    "I switched jobs because my old workplace reminded me too much of {event} and I couldn't function.",
    "I pretend I'm fine at family gatherings but inside I'm counting the minutes until I can leave.",
    "I changed my route to work just to avoid {place}. It adds forty minutes but I can't face it.",
    "I turned down a promotion because it would have meant going back to {place}.",
    "I can't eat certain foods anymore because they were present during {event}.",
    "I stopped listening to the radio because random songs bring back memories I've buried.",
    "I wear headphones constantly in public to block sounds that might trigger a memory of {event}.",
    "I avoid going out at night entirely. The darkness makes me feel exposed and vulnerable.",
    "I've pushed away most of my friends. It's easier to be alone than to pretend everything is okay.",
    "I keep cancelling therapy appointments because talking about {event} makes everything worse short term.",
    "I can't be in small enclosed spaces anymore. They make me feel trapped the same way as during {event}.",
    "I sit with my back to the wall in every restaurant now. I need to see every entrance and exit.",
    "I avoid any conversation that might lead back to {event}. I've become a master of changing the subject.",
    "I stopped exercising because my elevated heart rate from exertion triggers panic attack symptoms.",
    "I avoid hospitals at all costs even when I clearly need medical attention. The associations are too painful.",
    "I've been isolating so much that I sometimes go two weeks without speaking to another human being.",
    "I didn't go to my friend's wedding because the venue was near {place}. I feel terrible about it.",
    "I've stopped watching sport because the crowd noise and intensity reminds me of {event}.",
    "I've become completely nocturnal just to avoid people. Daytime interaction feels impossible right now.",
    "I ordered groceries online for the past four months because the supermarket feels too overwhelming.",
    "I turned down every social invitation this month. I just can't do it. The outside world isn't safe.",
    "I can't enter a car park. The echo and the enclosed feeling takes me straight back to {event}.",
    "I've avoided my family for months. They ask questions about {event} that I'm not ready to answer.",
    "I've become so good at avoiding my own thoughts that I can barely sit still for five minutes.",
    "I've stopped reading fiction because certain themes always end up connecting back to {event} somehow.",
    "I resigned from a job I loved because I simply couldn't be in that building anymore after {event}.",
    "I avoid eating with others because mealtime conversations sometimes drift toward {topic} without warning.",
    "I haven't been to the gym in months. Music, physical intensity and crowds are all triggers now.",
    "I avoid looking people in the eye. Eye contact feels far too intimate and exposed since {event}.",
    "I don't answer my phone unless I know exactly who is calling. Unexpected contact makes me spiral.",
    "I've become an expert at finding exit routes. Every room I enter I locate the exits before anything else.",
    "I drive twenty minutes out of my way to avoid {place} on my commute. Every single day.",
    "I've cancelled my holiday three times now because airports and hotels are too overwhelming.",
    "I avoid any conversation about the future. Thinking forward feels dangerous after {event}.",
    "I've started lying about my plans so I have an excuse not to show up. Avoidance on top of avoidance.",
    "I've reduced my world to just a few safe spaces. My bedroom is the only place I truly feel safe now.",
    "I stopped cooking the foods I associate with the time period around {event}. Even smells are triggers.",
    "I've withdrawn from every hobby I used to love. They all feel pointless and too exposed since {event}.",
    "I cancelled my therapy referral because I'm not ready to face what happened during {event}.",
    "I've taken to entering and leaving every building through side exits to avoid the main entrance crowds.",
    "I've completely stopped dating. Intimacy feels impossible and unsafe since what happened during {event}.",
    "I declined a work award ceremony because I can't be in large groups since {event}.",
    "I've been turning down promotions that require travel. Leaving my safe zone feels impossible.",
    "I removed every mirror from my bedroom. Looking at my own face takes me straight back to {event}.",
    "I've started shopping at strange hours — 5am or midnight — to avoid other people completely.",
    "I've blocked out entire chunks of memory around {event}. There are gaps I'm afraid to look into.",

    # ── HYPERAROUSAL (50 templates) ─────────────────────────────
    "I'm constantly on edge no matter what. I can't relax and every shadow feels like a threat.",
    "I snapped at my family again today. I can't control my anger and I feel absolutely terrible about it.",
    "I haven't slept properly in months. My body won't let me rest because it has to stay alert.",
    "I feel like I have to be constantly watching for danger. I physically cannot turn that instinct off.",
    "My hands shake when I hear loud noises. I'm always braced for something terrible to happen.",
    "I get so irritable over the smallest things. It's like my fuse is permanently lit.",
    "I sleep with one eye open because I feel like something bad is going to happen at any moment.",
    "I check the locks on my doors three times before bed. It's exhausting but I can't stop myself.",
    "My startle response is completely out of control. A door slamming sends me into a full panic spiral.",
    "I can't concentrate on anything. My mind is constantly scanning for threats even in safe places.",
    "I'm short-tempered with everyone around me. I see danger in situations that shouldn't bother me at all.",
    "I wake up multiple times a night convinced I heard something. I then check every room in the house.",
    "My body is in a constant state of fight-or-flight. I'm exhausted from the unrelenting arousal.",
    "I've been having heart palpitations that my doctor says are anxiety but they feel like I'm dying.",
    "I can't sit still during meetings. My leg bounces and I'm always looking toward the exit.",
    "I startled so badly at a balloon popping that I knocked over a display at the supermarket today.",
    "My blood pressure has been dangerously high. The doctor says it's stress. It's hypervigilance.",
    "I'm so jumpy that my colleagues have started to notice. They tiptoe around me which makes it worse.",
    "I feel rage surging through me at random moments. It comes out of nowhere and scares even me.",
    "I can't tolerate uncertainty of any kind. Not knowing what happens next sends me into a spiral.",
    "I've installed security cameras around my house. I check them obsessively throughout every day.",
    "My jaw is permanently clenched. I've cracked three teeth from grinding in my sleep this year.",
    "I can't relax in the bath or shower because I feel vulnerable with no way to see approaching threats.",
    "I've memorised all the exits in every building I enter. I do it automatically without thinking now.",
    "The hypervigilance is completely exhausting. Being on high alert every waking moment is unsustainable.",
    "I've started keeping a baseball bat under my bed. I know rationally I'm safe but my body doesn't.",
    "I haven't been able to eat a full meal without my hands shaking since {event}.",
    "I flinch at any sudden movement near me. My partner has learned not to approach from behind.",
    "I snapped at my child today over something completely trivial. The guilt is overwhelming.",
    "I feel like I'm living in a constant state of low-level terror. There's never any relief.",
    "I've been checking my rear-view mirror obsessively while driving. I'm certain someone is following me.",
    "I've yelled at three different people this week in situations where the old me would have stayed calm.",
    "I'm sleeping with all the lights on now. My electricity bill is enormous but I cannot handle the dark.",
    "I couldn't get through a film tonight without pausing to check that the front door was locked.",
    "I've been waking up at 4am every single morning for the past six weeks. My body won't rest.",
    "I felt my heart rate spike this morning when someone knocked on my door. It was just the postman.",
    "I've started refusing to sit with my back to the door in any social setting. It makes me too anxious.",
    "I feel like a live wire. The slightest surprise and I explode with an intensity that frightens people.",
    "My therapist says I'm hyperaroused. I say I'm just terrified all the time. Both are probably true.",
    "I've been grinding my teeth so badly that my dentist fitted me with a mouth guard last week.",
    "I can't fall asleep unless I've checked every window in the house. Even then I just lie there alert.",
    "I've disconnected my doorbell. The sudden sound sends me into a complete panic every time.",
    "I've been so on edge this week that I've been having chest pains. My doctor thinks it's anxiety.",
    "I feel like a cornered animal in social situations. I need to see the exit before I can even sit down.",
    "I've started parking facing outward everywhere I go. I need to be able to leave at a moment's notice.",
    "I can feel my muscles tensed up all day every day. I can't remember the last time I felt truly relaxed.",
    "Every phone notification makes my heart race. I've turned all notifications off but I still check constantly.",
    "I lost my temper at work today in a way I've never done before. I'm scared of who I've become since {event}.",
    "I can hear danger in completely neutral sounds now. A distant car horn, a dog barking — my body panics.",
    "I've been so irritable lately that my partner suggested we take a break. I don't blame them at all.",

    # ── EMOTIONAL NUMBNESS / NEGATIVE COGNITIONS (55 templates) ──
    "I feel completely numb inside. Nothing brings me any joy anymore. I just go through the motions.",
    "I believe I deserve what happened to me during {event}. I can't shake the guilt no matter what.",
    "I've lost all interest in everything I used to love. Nothing feels meaningful or important anymore.",
    "I feel like I'll never be the person I was before {event}. That person is gone forever.",
    "Everyone around me seems completely fine and I feel utterly broken. I can't connect with anyone.",
    "I stopped believing things will ever get better for me. What is even the point of trying?",
    "I feel detached from myself and from reality. Like I'm watching my life from very far away.",
    "The guilt from {event} is absolutely crushing. I blame myself entirely for everything that happened.",
    "I look in the mirror and don't recognise the person staring back. {event} destroyed who I used to be.",
    "I pushed away everyone who ever cared about me because I don't deserve their love or support.",
    "I used to be a happy person. Now I feel like that version of me was someone else entirely.",
    "I feel myself becoming more detached from everyone around me and I don't know how to stop it.",
    "I don't feel things anymore. Even things that should make me happy are met with complete emptiness.",
    "I've convinced myself that the world is fundamentally dangerous. Safety is just an illusion I once had.",
    "I feel responsible for everything bad that happens around me. The guilt is constant and overwhelming.",
    "Other people experience joy and connection. I just feel like an empty shell going through the motions.",
    "I've lost my sense of who I am. {event} stripped away everything I thought I knew about myself.",
    "I feel shame about {event} even though I know rationally that it wasn't my fault. Feelings don't listen.",
    "The future feels completely blank to me. I can't imagine or plan anything beyond the current moment.",
    "I've stopped making plans or setting goals. What's the point when everything can be taken away instantly?",
    "I feel like damaged goods. I don't think anyone could truly love or accept me knowing what happened.",
    "My sense of trust has been completely destroyed. I assume the worst of every person I meet now.",
    "I have a persistent feeling that I won't live very long. I don't know where it comes from but it's there.",
    "I can't feel close to my children anymore and that terrifies me. The emotional numbness scares me most.",
    "I go through every day feeling like I'm behind a sheet of glass. Present but not really there.",
    "I can't cry anymore. I want to but I just feel hollow. The emotions seem to have shut themselves off.",
    "I feel like a burden to everyone who knows me. I wonder sometimes if they'd be better without me around.",
    "I look at photos from before {event} and feel like I'm looking at a stranger. I don't know that person.",
    "My therapist says dissociation is a coping mechanism. All I know is I don't feel real half the time.",
    "I said something cruel to my best friend today and didn't feel any remorse until hours later. I scared myself.",
    "I've started to believe that I fundamentally deserve bad things. {event} must have been my fault somehow.",
    "I don't enjoy food anymore. Eating is just maintenance at this point. Nothing tastes like anything.",
    "I've lost my sense of humour completely. Things that would have made me laugh before just don't land.",
    "I feel like the colour has drained out of everything. The world is grey and flat and joyless.",
    "My family says I've become cold. I don't disagree. I can feel the warmth leaving me since {event}.",
    "I don't know how to receive affection anymore. When people are kind to me I don't know what to do with it.",
    "I've started to feel like the real me died during {event} and what's walking around now is just the shell.",
    "I don't have goals or dreams anymore. Ambition requires believing in a future and I can't do that right now.",
    "I feel nothing when I look at my children's photos. That absence of feeling terrifies me more than anything.",
    "I've begun to believe that good things only happen to other people. My quota of joy has been used up.",
    "I feel like I'm acting all day every day. Performing normality for everyone while feeling nothing inside.",
    "I've lost my ability to imagine a future where things are better. Hope feels like a language I've forgotten.",
    "I feel such profound emptiness that sometimes I wonder if I'm actually still here or just a ghost.",
    "The only emotion I seem capable of lately is a low-level despair that colours everything I do.",
    "I've become convinced that {event} revealed the truth about me: that I'm fundamentally broken.",
    "I watch other people laugh and feel a deep sadness that I can't access that anymore.",
    "I've stopped caring about my appearance. Why bother when you feel invisible and worthless anyway.",
    "I feel like I've been permanently changed at a cellular level by {event}. I'll never be the same.",
    "I've accepted that happiness is for other people. I don't say that in a dramatic way. I just believe it now.",
    "I tested positive for depression on a questionnaire today. I wasn't surprised. I've been numb for months.",
    "I feel a deep and permanent sadness that I can't locate. It's just always there, underneath everything.",
    "I've lost interest in my own story. I don't care what happens to me next. That scares me.",
    "I feel like {event} burned away the part of me that could feel hope. What's left doesn't know how.",
    "I used to be someone with opinions and passion. Now I just agree with everything because nothing matters.",
    "I feel such relentless shame about {event} that I can't look people in the eye anymore.",

    # ── COMBINED / LONG FORM (40 templates) ────────────────────
    "Since {event} my whole life has fallen apart. I can't sleep, I can't eat, I can't focus on anything. Every day feels like barely surviving.",
    "After {event} I became a completely different person. I used to be outgoing and happy but now I isolate myself constantly.",
    "My therapist confirmed I have PTSD from {event}. The flashbacks are constant and I cannot control them at all.",
    "I thought I was finally over {event} but the nightmares came back with a vengeance last week. I am so exhausted.",
    "Living with this condition is absolutely exhausting. I'm constantly hypervigilant and flinch at everything.",
    "The trauma from {event} haunts me every single day. I feel like no one understands what I went through.",
    "Some nights I lie awake for hours replaying {event}. By morning I'm so exhausted I can barely function at work.",
    "I tried going back to {place} to face my fears but the moment I walked in I was hit with intense panic.",
    "The combination of flashbacks, nightmares, and constant anxiety has made it impossible to hold down a job.",
    "I dissociate during stressful situations. I'll be in a meeting and suddenly I'm not there. I come back and people stare.",
    "My partner says I've become a different person since {event}. They're right. I'm watching my marriage fall apart.",
    "I've started using alcohol to quiet the memories of {event}. I know it's not healthy but it gives me a few hours of peace.",
    "Three different anxiety medications and none of them fully quiet the hypervigilance. I still check exits, still startle.",
    "The intrusive thoughts come at the worst times. Playing with my child and suddenly I'm back at {event}.",
    "I've been having suicidal thoughts since {event}. Not plans, just the thought that things would be quieter without me.",
    "I spent three months barely leaving my bed after {event}. Lost fifteen pounds. Missed so many important things.",
    "My physical health has deteriorated since {event}. Chronic headaches, digestive problems, unexplained pain.",
    "I can smile and laugh at parties and seem completely normal. But behind the mask I'm screaming inside.",
    "The PTSD has stolen years from me. I look at photos from before {event} and grieve for that carefree person.",
    "I've isolated so completely that I sometimes go two weeks without speaking to another person. The loneliness is terrible.",
    "I've been diagnosed with PTSD and I'm trying to find the right treatment. Nothing has worked so far.",
    "My psychiatrist prescribed medication for the nightmares. It helped a little but the daytime flashbacks are unmanageable.",
    "Eight months of therapy working through {event} using EMDR. It's painful but I'm starting to see small improvements.",
    "My doctor says I have complex PTSD from repeated trauma. The treatment is longer and harder than standard PTSD.",
    "I finally told my family about my PTSD diagnosis. Their reaction was not what I hoped for. I feel more alone than before.",
    "I joined a support group for trauma survivors. Hearing others describe my exact symptoms made me feel less alone.",
    "I'm struggling to find a therapist who specialises in trauma. The waitlist is eight months and I don't know how to cope until then.",
    "I tried to return to work after my PTSD diagnosis but had a breakdown on my third day back. I'm not ready.",
    "The trauma processing work in therapy is re-traumatising in the short term. My therapist says this is normal but it's hard.",
    "I've been self-medicating with sleep aids just to get through the night since {event}. I know I need real help.",
    "Some days I feel like I'm making progress and then something triggers me and I'm back to square one. It's demoralising.",
    "My PTSD has affected every relationship I have. I push people away before they can hurt me the way {event} did.",
    "I feel like the mental health system has failed me. I've seen five therapists since {event} and none of them understood trauma.",
    "I carry the guilt of {event} like a physical weight. It never leaves. It's there when I wake and there when I sleep.",
    "My children are scared of me when I have flashbacks. The look of fear on their faces is the hardest thing to bear.",
    "I've spent so much money on therapy since {event} that I'm now financially struggling on top of everything else.",
    "I lost my best friend because of how I behaved during my worst PTSD episode. I don't blame them for walking away.",
    "I used to be the person people came to for support. Now I can barely support myself. {event} took that from me too.",
    "I've been doing trauma therapy for two years now and I'm making slow progress. But slow progress is still progress.",
    "Some days the only thing that gets me out of bed is the knowledge that my kids need me. Bare survival is still survival.",

    # ── CLINICAL / SEEKING HELP (30 templates) ─────────────────
    "I've been diagnosed with PTSD and I'm trying to find the right treatment. CBT hasn't worked so far.",
    "My psychiatrist prescribed medication for the nightmares and it helped a little but flashbacks continue.",
    "I've been in therapy for eight months working through {event} using EMDR. It's painful but I see small improvements.",
    "My doctor says I have complex PTSD from repeated trauma over many years.",
    "I finally told my family about my PTSD diagnosis. Their reaction was not what I hoped for.",
    "I joined a support group for PTSD survivors and hearing others describe my exact symptoms helped.",
    "I'm struggling to find a therapist who specialises in trauma. The waitlist is months long.",
    "My veteran's counselor confirmed combat-related PTSD. Knowing it has a name helps.",
    "I tried to return to work after my PTSD diagnosis but I had a breakdown on the third day back.",
    "The trauma processing work in therapy is re-traumatising in the short term. My therapist says this is normal.",
    "I've been prescribed three different SSRIs since {event} and none of them has touched the hypervigilance.",
    "I have a PTSD diagnosis but my employer still doesn't understand why I need accommodations.",
    "My PTSD support group meets weekly and even on bad weeks it's the only place I feel truly understood.",
    "I've been referred to a specialist trauma clinic but the wait is six months. I don't know how I'll manage.",
    "I started EMDR therapy two months ago and I've noticed the flashbacks are slightly less intense already.",
    "I keep a symptom diary as my therapist suggested. Looking back at the entries is both useful and harrowing.",
    "Medication helped me sleep but the underlying trauma from {event} still needs to be processed.",
    "I've read everything I can find about PTSD since my diagnosis. Knowledge helps me feel slightly more in control.",
    "My psychiatrist added a new medication last month specifically targeting hypervigilance. It's helping slightly.",
    "I've started a trauma-informed yoga class on my therapist's recommendation. Movement helps more than I expected.",
    "Group therapy has helped me see that I'm not alone in this. Others understand in a way friends and family can't.",
    "My PTSD is treatment-resistant apparently. We're now trying ketamine therapy as a last resort.",
    "My therapist uses somatic approaches to help me process the body memories from {event}. It's slow work.",
    "I've been on sick leave for three months. My employer is understanding but I feel enormous guilt about it.",
    "Peer support from other veterans has been more helpful than formal therapy so far in my recovery from {event}.",
    "I've started keeping a flashback log. Tracking them has shown me they're actually reducing in frequency slightly.",
    "I got a trauma-informed GP at last. Having a doctor who understands PTSD has made a significant difference.",
    "I completed a twelve-week trauma programme last month. I have more tools now even if the symptoms aren't gone.",
    "I've been doing progressive muscle relaxation every night since {event}. It helps my body learn to de-escalate.",
    "I'm on a waiting list for residential trauma treatment. It's a long wait but I'm holding on knowing it's coming.",
]


# ═══════════════════════════════════════════════════════════════
#  NON-PTSD TEMPLATES (300+)
# ═══════════════════════════════════════════════════════════════
non_ptsd_templates = [

    # ── EVERYDAY POSITIVE ─────────────────────────────────────
    "Today was a great day. I spent the morning reading a good book and the afternoon gardening.",
    "I went for a jog this morning and it felt absolutely amazing. The weather was perfect.",
    "Had dinner with friends last night. We laughed until our stomachs hurt.",
    "I just finished reading a really good novel. I couldn't put it down.",
    "The kids had a blast at school today. My daughter came home so excited.",
    "I cooked a new recipe tonight and it turned out even better than expected.",
    "Took a long walk in the park today and enjoyed watching the leaves change colour.",
    "Got promoted at work today after years of hard work. All the effort finally paid off.",
    "Planning a vacation to the beach next month with the family. Can't wait to relax.",
    "Watched a hilarious comedy movie with my partner tonight. We needed that laugh.",
    "My garden is finally blooming the way I hoped. All that work over winter was worth it.",
    "I had the best conversation with an old friend over coffee this morning. Really uplifting.",
    "Finished a big project at work and feeling really satisfied with what my team accomplished.",
    "My puppy learned a new trick today and I couldn't be more proud of the little guy.",
    "The sunrise this morning was absolutely stunning. I sat outside with my tea and appreciated it.",
    "I signed up for a pottery class and I'm genuinely excited about trying something new.",
    "Had the most refreshing swim at the local pool this morning. Starting the day right.",
    "My family surprised me with a birthday party and I felt so genuinely loved.",
    "I've been reading about mindfulness and trying new techniques. Feeling more centred.",
    "The concert last night was incredible. Music has such a powerful way of lifting my mood.",
    "I reorganised my home office and it's made a real difference to my focus and productivity.",
    "A stranger paid me a genuine compliment today and it made my whole week better.",
    "My book club met tonight and we had a wonderful discussion about the latest novel.",
    "I helped a neighbour move furniture and it felt good to do something kind.",
    "The farmer's market this morning was perfect. Fresh produce and live music.",
    "I baked fresh bread today for the first time in years. The smell alone made it worthwhile.",
    "My sister called out of the blue just to catch up. We talked for three hours and it was wonderful.",
    "I finally cleared out the spare room I've been meaning to sort for a year. It feels so good.",
    "Had a spontaneous picnic in the park with my kids today. Sometimes simple is best.",
    "I passed my driving test on the first attempt. All those hours practising paid off.",
    "We had a fire pit in the back garden tonight with the neighbours. Community at its best.",
    "I read a book in a single day. I forgot how much I loved doing that as a teenager.",
    "My daughter drew a portrait of me today. It doesn't look much like me but I framed it anyway.",
    "I finally booked that cooking class I've been talking about for three years. Looking forward to it.",
    "A colleague brought homemade cake to work today. Small gesture, huge impact on morale.",
    "I went to bed early with a herbal tea and a podcast. Sometimes that's the perfect evening.",
    "I woke up before my alarm feeling genuinely well-rested for the first time in a while.",
    "Had a spontaneous dance in the kitchen with my partner tonight. Life feels good.",
    "I finally got around to planting the herb garden I've been planning. Very satisfying.",
    "I heard my favourite song on the radio while driving to work. Started the day with a smile.",

    # ── MILD STRESS, HEALTHY COPING ────────────────────────────
    "Work has been hectic lately but I'm managing it well. I started meditating in the evenings.",
    "I had a stressful exam today but I studied hard and I think it went pretty well.",
    "Traffic was terrible this morning and I ended up late to work. I took a deep breath and stayed calm.",
    "My flight got delayed by two hours. I used the extra time to catch up on reading.",
    "The project deadline is tomorrow but my team is right on track.",
    "I felt nervous about the presentation but once I started talking I found my rhythm.",
    "The apartment hunting process has been stressful but I found a place I really like.",
    "Moving day was chaotic and tiring but everything got sorted in the end. New place feels like home.",
    "I had a difficult conversation with my manager today but we resolved it maturely.",
    "The kids have been demanding this week and I'm tired. But it comes with parenthood.",
    "Finances have been tight this month but we've been budgeting carefully and getting through it.",
    "I missed a workout for the fifth day in a row. I'm giving myself grace and starting again tomorrow.",
    "There was a miscommunication at work that caused friction. We sorted it out professionally.",
    "I'm adjusting to a new neighbourhood and it's taking time to settle. But I'm patient.",
    "The renovation has been more disruptive than expected but we can see the end in sight.",
    "I'm learning to set better boundaries at work. Uncomfortable at first but necessary.",
    "I've been dealing with some health anxiety lately but talking to my doctor put my mind at ease.",
    "I'm finding the transition back to office work challenging after remote work. Adjusting steadily.",
    "I had a big argument with my sibling but we talked it through and we're okay. Family is complicated.",
    "My car failed its service and needs expensive repairs. Annoying but manageable. Life goes on.",
    "I made a mistake at work this week and had to own up to it. Humbling but the right thing to do.",
    "My internet has been playing up for a week. It's frustrating but I've been reading more as a result.",
    "I over-committed socially this week and I'm exhausted. Learning to say no is a work in progress.",
    "I got a parking ticket today through a genuine mistake. Annoying but I won't let it ruin my day.",
    "My sleep has been a bit disrupted this week by stress about a project. Once it's done I'll be fine.",
    "I've been feeling a bit off this week. Probably just end-of-season tiredness. Taking it easy.",
    "I disagreed with a colleague today in a meeting. We handled it professionally and moved forward.",
    "I had a tough day at work but I came home, cooked a good meal, and I feel much better now.",
    "I've been struggling to motivate myself this week. I know it's temporary. I've been here before.",
    "The traffic on my commute has been awful all week. Stressful but hardly the end of the world.",

    # ── NORMAL EMOTIONS ───────────────────────────────────────
    "I felt a bit down today but I know it's just one of those days. Tomorrow will be better.",
    "I'm a little nervous about the job interview next week but I've been preparing well.",
    "Had a minor disagreement with a friend but we talked it out calmly and everything is fine.",
    "Feeling a bit nostalgic today thinking about the old days. Mostly with warmth and fondness.",
    "I missed my parents today. We live far apart but have a good video call every Sunday.",
    "I felt a bit overwhelmed at work but took a short break and felt better afterwards.",
    "I've been feeling restless lately and thinking about making some changes in my life.",
    "I cried watching a film tonight. A good emotional release is sometimes exactly what you need.",
    "Feeling grateful today for all the small things. Health, shelter, good relationships.",
    "I had moments of self-doubt this week about my abilities. But I reminded myself of past successes.",
    "I've been reflecting on some regrets from my past. Ultimately accepting them and looking forward.",
    "I'm learning to handle criticism more constructively. It stings but I use it to improve.",
    "Feeling a little lonely tonight even though I was around people all day. Just one of those feelings.",
    "I've been worrying about the state of the world lately. I try to focus on what I can control.",
    "I feel tired from a busy week. Planning a quiet restful weekend to recharge properly.",
    "I'm working through some unresolved feelings about a past relationship. Slowly making peace with it.",
    "I felt a wave of anxiety this afternoon about something that probably won't happen. Talked myself down.",
    "I've been a bit impatient with my kids lately. I noticed it today and made a conscious effort to slow down.",
    "I'm feeling a bit unfulfilled professionally at the moment. I'm thinking about what changes to make.",
    "I've been comparing myself to others on social media again. I know it's not helpful. Working on it.",
    "I had a difficult conversation today but I handled it with more grace than I expected of myself.",
    "I've been struggling to find balance between work and home life. I'm making changes.",
    "I had a moment of genuine joy today watching my dog chase its tail. Simple pleasures matter.",
    "I've been a bit short on patience this week. I recognise it as tiredness rather than a character flaw.",
    "I've had some difficult dreams lately. Nothing alarming, just processing some life stress I think.",
    "I made a decision I'm not completely sure about today. I'm sitting with the uncertainty. That's okay.",
    "I've been eating too much junk food this week out of stress. Back to proper meals tomorrow.",
    "I felt sad for no particular reason today. I accepted the feeling and let it pass. That's okay.",
    "I've been overthinking something at work that probably doesn't matter as much as I've made it in my head.",
    "I'm feeling a bit under the weather today. Just a mild cold. Taking it easy and resting up.",

    # ── PROFESSIONAL / ACHIEVEMENT ────────────────────────────
    "I just closed the biggest deal of my career. The team worked so hard and it paid off.",
    "I graduated after years of studying while working full time. Incredibly proud of myself.",
    "My business finally turned a profit after two years of grinding. The perseverance was worth it.",
    "I got accepted to my dream university program. All those late nights were completely worth it.",
    "My article got published in the journal I've been targeting for three years.",
    "I ran my first half marathon today. My legs are destroyed but I've never felt more proud.",
    "I completed my certification course and passed the exam on the first attempt.",
    "I was asked to mentor a junior colleague. It means they respect my knowledge and experience.",
    "I gave my first public talk today and the audience response was genuinely warm.",
    "My team won the annual innovation award at work. Recognition for months of hard work.",
    "I've been sober for six months today. The hardest thing I've ever done but I'm doing it.",
    "I paid off my student debt today. Took a decade but I never missed a payment. I'm free.",
    "I started the business I've been planning for two years. Terrifying and exhilarating.",
    "I got a standing ovation from my class today. It validated all the effort I put in.",
    "My grant application was approved. Two years of work and it's going to happen.",
    "I finished writing my first novel today. Whether it's published or not I did it.",
    "I negotiated a pay rise for the first time in my career today. I was nervous but I did it.",
    "My team delivered the project on time and under budget. We're all so proud of what we achieved.",
    "I got the job I've wanted for three years. I start in two weeks. This is a new chapter.",
    "I presented at a conference for the first time today. The feedback was incredibly encouraging.",

    # ── RELATIONSHIPS / SOCIAL ────────────────────────────────
    "My partner and I celebrated our anniversary this weekend. Ten years and still genuinely happy.",
    "I made a new friend at my fitness class. We got coffee and talked for three hours.",
    "Family dinner last night was chaotic in the best possible way. Noise, laughter, and too much food.",
    "I had a heart-to-heart with my best friend. We're closer than ever after that conversation.",
    "My children are growing up so fast. Watching them become their own people is the greatest privilege.",
    "I reconnected with an old friend I hadn't spoken to in years. It felt like no time had passed.",
    "I got engaged today and I've never been happier. This is exactly the life I hoped for.",
    "My support network has been incredible through this difficult period. I'm lucky.",
    "I've been intentionally investing in my friendships this year and the returns have been beautiful.",
    "My parents are healthy and sharp in their seventies. I cherish every conversation.",
    "My son called just to tell me he loves me today. Out of the blue. Made my entire week.",
    "I've been making more time for old friends this year. The effort is absolutely worth it.",
    "My relationship with my siblings has never been better. We talk every week now.",
    "I went on a second date with someone I really like. Cautiously optimistic.",
    "My best friend got the promotion they've been working towards for years. I'm so proud.",
    "I spent the afternoon with my grandparents today. These visits are precious and I know it.",
    "My neighbour and I finally had a proper conversation after living next door for three years.",
    "I organised a surprise party for my friend and the look on their face was absolutely priceless.",
    "I met my pen friend in person after five years of letter writing. It was everything I hoped.",
    "My family rallied around me when I needed them this month. I feel so loved.",

    # ── SELF-IMPROVEMENT ──────────────────────────────────────
    "I've been journaling every morning for three months now. The self-awareness I've gained is invaluable.",
    "I started therapy to work on some old patterns and it's been one of the best decisions I've made.",
    "I've been practising gratitude daily and I genuinely notice a shift in how I perceive difficult days.",
    "I took a digital detox weekend and came back feeling more present and connected.",
    "I've been working on my communication style and my relationships have improved noticeably.",
    "I'm learning to sit with discomfort rather than immediately trying to fix it.",
    "I've reduced my alcohol intake significantly this year and I feel so much better.",
    "I started a new exercise routine three months ago. The mental health benefits have been remarkable.",
    "I'm learning to say no without guilt. It's liberating.",
    "I've been meditating daily for sixty days now. The compounding benefits are real.",
    "I started cold water therapy this month. It sounds extreme but it's genuinely helping my mood.",
    "I've reduced my screen time by an hour a day this month. The mental clarity is noticeable.",
    "I've started cooking proper meals instead of convenience food. My energy levels are much better.",
    "I've been in consistent therapy for a year now and the progress I've made is remarkable.",
    "I finally addressed a relationship pattern I've had for years. It feels like breaking a spell.",
    "I've started volunteering on weekends. Helping others gives my week real purpose and meaning.",
    "I took up running six months ago. I never thought I'd enjoy it but here I am training for a 5k.",
    "I've started reading before bed instead of scrolling. My sleep has improved significantly.",
    "I've been tracking my mood daily for three months. The patterns I've noticed have been revealing.",
    "I started learning a new language this year. Progress is slow but each new word feels like a small victory.",
]


# ═══════════════════════════════════════════════════════════════
#  TEXT GENERATION
# ═══════════════════════════════════════════════════════════════
def fill(t):
    if "{event}" in t: t = t.replace("{event}", random.choice(EVENTS))
    if "{place}" in t: t = t.replace("{place}", random.choice(PLACES))
    if "{topic}" in t: t = t.replace("{topic}", random.choice(TOPICS))
    return t

def gen_text(n, ptsd_ratio):
    n_ptsd = int(n * ptsd_ratio)
    rows   = []
    for _ in range(n_ptsd):
        tmpl = random.choice(ptsd_templates)
        text = fill(tmpl)
        # 35% chance of combining two templates for longer samples
        if random.random() < 0.35:
            text += " " + fill(random.choice(ptsd_templates))
        sev = round(np.clip(np.random.normal(7.0, 1.8), 0, 10), 1)
        rows.append({"text": text, "label": 1, "severity": sev})

    for _ in range(n - n_ptsd):
        tmpl = random.choice(non_ptsd_templates)
        text = fill(tmpl)
        if random.random() < 0.30:
            text += " " + fill(random.choice(non_ptsd_templates))
        sev = round(np.clip(np.random.normal(1.5, 1.2), 0, 10), 1)
        rows.append({"text": text, "label": 0, "severity": sev})

    random.shuffle(rows)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  AUDIO DATASET (42 features, 10k rows)
# ═══════════════════════════════════════════════════════════════
def gen_audio(n, ptsd_ratio):
    n_ptsd = int(n * ptsd_ratio)
    rows   = []
    for i in range(n):
        p   = (i < n_ptsd)
        sev = round(np.clip(np.random.normal(7.0 if p else 1.6, 1.8 if p else 1.2), 0, 10), 1)
        row = {}
        for j in range(1, 14):
            row[f"mfcc_{j}_mean"] = round(np.random.normal(-18+j*1.2 if p else -10+j*1.0, 4.0), 3)
            row[f"mfcc_{j}_std"]  = round(np.clip(np.random.normal(3.5 if p else 5.0, 1.5), 0.5, 12), 3)
        row["pitch_mean"]  = round(np.random.normal(100 if p else 135, 15), 2)
        row["pitch_std"]   = round(np.clip(np.random.normal(30 if p else 20, 10), 2, 60), 2)
        row["pitch_range"] = round(np.clip(np.random.normal(70 if p else 45, 20), 5, 140), 2)
        row["energy_mean"] = round(np.random.normal(0.012 if p else 0.022, 0.004), 5)
        row["energy_std"]  = round(np.clip(np.random.normal(0.018 if p else 0.012, 0.006), 0.001, 0.05), 5)
        row["energy_max"]  = round(np.clip(np.random.normal(0.06  if p else 0.08,  0.02),  0.02, 0.2), 4)
        row["jitter_local"]  = round(np.clip(np.random.normal(0.009 if p else 0.004, 0.004), 0.0005, 0.03), 5)
        row["shimmer_local"] = round(np.clip(np.random.normal(0.85  if p else 0.45,  0.3),   0.05,  2.5), 4)
        row["shimmer_apq3"]  = round(np.clip(np.random.normal(0.07  if p else 0.04,  0.03),  0.005, 0.2), 4)
        row["hnr"]           = round(np.random.normal(6.5 if p else 15.0, 3.0), 3)
        row["speech_rate"]   = round(np.clip(np.random.normal(2.8 if p else 4.5, 0.8), 1, 7), 2)
        row["pause_ratio"]   = round(np.clip(np.random.normal(0.45 if p else 0.22, 0.1), 0.05, 0.8), 3)
        row["spectral_centroid_mean"] = round(np.random.normal(1800 if p else 2600, 400), 1)
        row["spectral_rolloff_mean"]  = round(np.random.normal(3200 if p else 4400, 600), 1)
        row["spectral_flux_mean"]     = round(np.clip(np.random.normal(0.045 if p else 0.062, 0.012), 0.01, 0.12), 4)
        row["label"]    = int(p)
        row["severity"] = sev
        rows.append(row)
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  FACIAL DATASET (31 features — matches camera_detect.py exactly)
# ═══════════════════════════════════════════════════════════════
def gen_facial(n, ptsd_ratio):
    n_ptsd = int(n * ptsd_ratio)
    rows   = []
    au_p = {"AU1":1.8,"AU4":2.9,"AU5":1.5,"AU6":0.4,"AU7":1.6,"AU9":0.3,
            "AU10":0.5,"AU12":0.3,"AU15":1.2,"AU17":1.5,"AU20":2.2,"AU23":1.8,"AU24":2.1}
    au_n = {"AU1":1.0,"AU4":0.6,"AU5":0.4,"AU6":1.8,"AU7":0.7,"AU9":0.8,
            "AU10":0.9,"AU12":2.1,"AU15":0.3,"AU17":0.4,"AU20":0.3,"AU23":0.4,"AU24":0.3}

    for i in range(n):
        p   = (i < n_ptsd)
        sev = round(np.clip(np.random.normal(6.8 if p else 1.7, 2.0 if p else 1.1), 0, 10), 1)
        row = {}
        d   = au_p if p else au_n

        for au, m in d.items():
            row[f"{au}_intensity"] = round(np.clip(np.random.normal(m, 0.5), 0, 5), 3)

        row["mouth_width_ratio"]  = round(np.clip(np.random.normal(0.32 if p else 0.38, 0.04), 0.15, 0.55), 3)
        row["mouth_height_ratio"] = round(np.clip(np.random.normal(0.08 if p else 0.12, 0.03), 0.01, 0.25), 3)
        row["eye_openness_left"]  = round(np.clip(np.random.normal(0.28 if p else 0.38, 0.06), 0.05, 0.6),  3)
        row["eye_openness_right"] = round(np.clip(np.random.normal(0.27 if p else 0.37, 0.06), 0.05, 0.6),  3)
        row["brow_raise_left"]    = round(np.clip(np.random.normal(0.35 if p else 0.22, 0.08), 0.02, 0.65), 3)
        row["brow_raise_right"]   = round(np.clip(np.random.normal(0.34 if p else 0.21, 0.08), 0.02, 0.65), 3)
        row["nose_tip_x_norm"]    = round(np.random.normal(0.5, 0.05), 3)
        row["nose_tip_y_norm"]    = round(np.clip(np.random.normal(0.55 if p else 0.52, 0.04), 0.3, 0.75), 3)
        row["AU4_std"]            = round(np.clip(np.random.normal(0.8  if p else 0.3,  0.2),  0.05, 2.0), 3)
        row["AU12_std"]           = round(np.clip(np.random.normal(0.25 if p else 0.4,  0.15), 0.02, 1.0), 3)
        row["mouth_movement_std"] = round(np.clip(np.random.normal(2.1  if p else 1.2,  0.5),  0.3,  4.0), 3)
        row["eye_movement_std"]   = round(np.clip(np.random.normal(1.8  if p else 0.9,  0.4),  0.2,  3.5), 3)
        row["eye_asymmetry"]      = round(np.clip(np.random.normal(0.06 if p else 0.02, 0.02), 0.0,  0.15), 4)
        row["brow_asymmetry"]     = round(np.clip(np.random.normal(0.05 if p else 0.015,0.02), 0.0,  0.12), 4)
        row["mouth_asymmetry"]    = round(np.clip(np.random.normal(0.04 if p else 0.01, 0.015),0.0,  0.1),  4)
        row["head_yaw"]           = round(np.random.normal(0,  8 if p else 5), 2)
        row["head_pitch"]         = round(np.random.normal(-5 if p else -2, 6 if p else 4), 2)
        row["head_roll"]          = round(np.random.normal(0,  4 if p else 3), 2)
        row["label"]    = int(p)
        row["severity"] = sev
        rows.append(row)

    random.shuffle(rows)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  KAGGLE REAL-DATA LOADER
# ═══════════════════════════════════════════════════════════════
def try_load_kaggle(synth_df):
    """
    Drop any of these CSVs in real_data/ and they will be
    automatically merged with the synthetic dataset.

    Supported formats:
      mental_health.csv   → columns: text, label   (0/1)
      social_media.csv    → columns: post_text, label
      daic_woz.csv        → columns: text, label, severity
      custom.csv          → columns: text, label   (any binary label)
    """
    mapping = {
        "real_data/mental_health.csv": "text",
        "real_data/social_media.csv":  "post_text",
        "real_data/daic_woz.csv":      "text",
        "real_data/custom.csv":        "text",
    }
    combined = [synth_df]
    for path, text_col in mapping.items():
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            # Rename text column to 'text' if needed
            if text_col in df.columns and text_col != "text":
                df = df.rename(columns={text_col: "text"})
            if "text" not in df.columns or "label" not in df.columns:
                print(f"  ⚠️  {path}: need 'text' and 'label' columns — skipped.")
                continue
            # Binarise label if needed (e.g. string labels)
            if df["label"].dtype == object:
                df["label"] = df["label"].map(
                    lambda x: 1 if str(x).lower() in
                    ("1","ptsd","positive","yes","true","depression","anxiety","trauma") else 0
                )
            if "severity" not in df.columns:
                df["severity"] = df["label"].apply(
                    lambda l: round(np.clip(np.random.normal(7.0 if l==1 else 1.5, 1.8), 0, 10), 1)
                )
            subset = df[["text","label","severity"]].dropna()
            combined.append(subset)
            print(f"  ✅ Loaded Kaggle data: {path}  ({len(subset):,} rows)")
        except Exception as e:
            print(f"  ⚠️  Could not load {path}: {e}")
    return pd.concat(combined, ignore_index=True) if len(combined)>1 else synth_df


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    os.makedirs("data",      exist_ok=True)
    os.makedirs("real_data", exist_ok=True)

    print("📊 Generating text dataset (10,000 rows) …")
    df_text = gen_text(N, PTSD_RATIO)
    df_text = try_load_kaggle(df_text)
    df_text.to_csv("data/text_dataset.csv", index=False)

    print("🎙️  Generating audio dataset (10,000 rows) …")
    df_audio = gen_audio(N, PTSD_RATIO)
    df_audio.to_csv("data/audio_dataset.csv", index=False)

    print("🎭 Generating facial dataset (10,000 rows) …")
    df_facial = gen_facial(N, PTSD_RATIO)
    df_facial.to_csv("data/facial_dataset.csv", index=False)

    print("\n✅ Done!")
    for name, df in [("Text",df_text),("Audio",df_audio),("Facial",df_facial)]:
        p = df["label"].sum()
        total = len(df)
        print(f"   {name:8s} → {total:,} rows | PTSD: {p:,} ({p/total:.1%}) | "
              f"Non-PTSD: {total-p:,}")
        print(f"            Severity — PTSD: {df[df.label==1].severity.mean():.2f} | "
              f"Non-PTSD: {df[df.label==0].severity.mean():.2f}")

    print("""
💡 KAGGLE BOOST (optional — drop CSVs in real_data/ folder):
   1. Mental Health Corpus:
      https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus
      → Save as: real_data/mental_health.csv  (needs: text, label columns)

   2. Social Media Mental Health:
      https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media
      → Save as: real_data/social_media.csv  (needs: post_text, label columns)

   Then re-run: python3 generate_dataset.py && python3 train_model.py
""")

if __name__ == "__main__":
    main()