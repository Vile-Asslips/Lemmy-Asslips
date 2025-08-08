UPDATE 08/08/2025.

USE VERY_DYNAMIC2.PY. It's the latest version!

Run with the line:

python very_dynamic2.py config.yaml

=============================
=============================

Best viewed in Notepad++


==============================================================================
==============================================================================
==============================================================================
==============================================================================
very_dynamic.py code stuff.
==============================================================================
==============================================================================
==============================================================================
==============================================================================


[pool] collects all the relevant keywords (title/body for posts; comment keywords for replies).

[dynamic_topics] becomes a natural English list like "battery, cartridge, and voltage".

[prompt] = bot_fp.format(topics=dynamic_topics) seeds your grotesque template with the entire list.

[self._gen(...)] then spins off a fresh, on-topic—but still hilariously disgusting—fallback reply.


==============================================================================
==============================================================================
==============================================================================
==============================================================================

CONFIG.YAML EXAMPLES:

==============================================================================
==============================================================================
==============================================================================
==============================================================================

Replace each bot’s fallback_prompt with one (or more) of these examples that use {topics}:

----------------------------------------------

bots:
  - name: Turbo
    …
    fallback_prompt: "I’ve just bathed my Asslips in a vat of rancid pus, mold, and unholy goo extracted from {topics}—here’s what I think…"

  - name: Dingleberry
    …
    fallback_prompt: "My brain’s a septic tank of dingleberries, zombie worms, and maggot-filled sludge born from {topics}—let’s dive in…"

  - name: SourAsslips
    …
    fallback_prompt: "I’m marinating in an inferno of butt-butter, festering sores, and pure rot distilled from {topics}. Now, here’s a hot take:"
	
	
----------------------------------------------

==============================================================================

Feel free to tweak the lead-in (“I’ve just…,” “My brain’s…,” “I’m marinating…”) or the trailing segue (“—here’s what I think…,” “—let’s dive in…,” “Now, here’s a hot take:”) to fit each bot’s personality.

==============================================================================
==============================================================================
==============================================================================
==============================================================================

More template examples

==============================================================================
==============================================================================
==============================================================================
==============================================================================

Plural-sensitive

fallback_prompt: "After inhaling the putrid fumes of {topics}, my Asslips are still buzzing—here’s a raw breakdown:"

==============================================================================
==============================================================================

Singular-style

fallback_prompt: "That {topics} swirl just birthed a garden of toxic butt-butter. Let’s untangle it:"


==============================================================================
==============================================================================

Mixed

fallback_prompt: "I soaked my privates in a drench of maggot-infested {topics}. Brace yourself for the aftermath:"

==============================================================================
==============================================================================


You can leave fallback_prompt blank or omit it on bots that should continue to use the YAML‐pool (fallbacks_expanded.yaml).
