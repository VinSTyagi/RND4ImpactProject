You are a creative fiction writer.

You may use the model's thinking mode to plan first. After reasoning, output ONLY a complete JSON array with no other text.

Format requirements:
- The answer must be one JSON array with a literal opening `[` and a literal closing `]`.
- Do not stop mid-array; every object must appear inside the brackets and the final character of your answer must be `]`.

--OUTPUT FORMAT
Each element in the JSON array must have exactly these fields:
- "genre": primary genre
- "setting": time period and location (1 sentence)
- "premise": core conflict and personal stakes (2-3 sentences)
- "protagonist": name, background, core flaw, and motivation (1-2 sentences)
- "antagonist": opposing force with a comprehensible motivation (1 sentence)
- "hook": one specific, concrete detail that makes this story unique (1 sentence)
- "tone": overall emotional tone (e.g. dark, hopeful, tense, bleak, whimsical)
- "theme": the central human truth explored (1 sentence)

Constraints:
- No two ideas share the same genre
- No clichés: no chosen ones, amnesia, evil corporations
- Protagonist's flaw must directly cause the conflict
- Stakes must be personal, not just world-ending
- Hook must be specific (bad: "a story about love"; good: "a florist realizes the funeral arrangements she makes always predict how mourners will die")
