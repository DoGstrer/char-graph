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

function CharacterRelationships(txt: string) -> Characters[] {
  client CustomGPT5Nano
  prompt #"
  Given the following text, identify the main characters and describe their relationships with each other.
  {{txt}}
  {{ctx.output_format}}
  "#

}