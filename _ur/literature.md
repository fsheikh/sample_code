---
layout: default
lang: UR
title: Literature
---

{% for tag in site.tags %}
  {% if tag[0] == "urdu literature" %}
  <ul>
    {% for post in tag[1] %}
      <li style="direction:rtl;font-size:24px;"><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}
{% endfor %}