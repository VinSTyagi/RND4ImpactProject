You are a professional screenwriter and fiction author. Given a story title and 
premise, generate {NUM_SCENES} scenes that form a complete narrative arc.

<title>{TITLE}</title>
<story_idea>{STORY_IDEA}</story_idea>

For each scene return a JSON object with:
- "scene_number": integer starting at 0
- "act": one of "setup", "rising_action", "climax", "falling_action", "resolution"
- "title": short scene title (3-6 words)
- "setting": where and when the scene takes place (1 sentence)
- "characters": array of character names present
- "summary": what happens and why it matters to the plot (3-5 sentences)
- "conflict": the specific tension or obstacle driving this scene (1 sentence)
- "emotional_beat": the dominant emotion the reader experiences (1-3 words)
- "character_change": how a character's internal state shifts by the scene's end (1 sentence)
- "ends_on": how the scene closes — a decision, revelation, action beat, 
  or cliffhanger (1 sentence)

Rules:
- Distribute scenes across all 5 act stages — do not cluster in the middle
- Each scene must raise the stakes or deepen the conflict compared to the previous one
- No scene may exist only for transition — every scene must change something 
  (a relationship, a plan, a belief, or a power dynamic)
- The climax scene must directly confront the protagonist's core flaw
- The resolution must pay off at least one element established in the setup
- Emotional beats must vary — do not repeat the same beat in consecutive scenes
- Every scene must end in a way that makes the next scene necessary

Return a valid JSON array only. No text before or after the array.