{{ title }}
{% for _ in title %}={% endfor %}

.. include:: ../../../../usersguide/source/changes/{{ title|lower()|replace(' released', '')|replace('.', '_')|replace(' ', '_')}}.rst
{{ content }}

.. author:: {{ author }}
.. tags:: release
.. comments::
