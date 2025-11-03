enum Weight{
  ACQUAINTANCE
  OUTER_CIRCLE
  INNER_CIRCLE
}

class Relationship{
  name string
  type string @description(#"The type of relationship, e.g., friendship, rivalry, mentorship."#)
  weight Weight @description(#"strength of relationship"#)
}

class Characters{
  name string
  role string @description(#"The role of the character in the story, e.g., protagonist, antagonist, sidekick."#)
  relationships Relationship[] @description(#"A list of relationships the character has with other characters in the story."#)
}

function CharacterRelationships(txt: string, prev_output: Characters[]) -> Characters[] {
  client CustomGPT5Nano
  prompt #"
    {% if not prev_output %}
    Given the following text, identify the main characters and describe their relationships with each other.
    <text>{{txt}}</text>
    {% else %}
    Given the following text and previous output, identify the main characters and describe their relationships with each other and output updated previous output.
    <text>{{txt}}</text>
    <previous>{{prev_output}}</previous>
    {% endif %}
    {{ctx.output_format}}
  "#

}