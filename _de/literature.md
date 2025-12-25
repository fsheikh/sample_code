---
layout: default
lang: DE
title: Literature
permalink: /_de/literature
---

- Shafay Sheikh: [Dam-di-Dum](https://www.landschreiber-wettbewerb.de/assets/texte/LSW%2011/Shafay%20Sheikh%20-%20dam-di-dum.pdf) aus [Landschreiber-Wettbewerb 11](https://www.landschreiber-wettbewerb.de/texte-ls.html)

<p></p>
{% for tag in site.tags %}
  {% if tag[0] == "german literature" %}
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}