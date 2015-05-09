{{ title }}
{% for _ in title %}={% endfor %}

.. include:: ../../../highlights/{{ title | lower() | replace(' released', '') |
replace('.', '_') | replace(' ', '_') }}.rst.inc

Download this release from the :ref:`download section <download>`.

..
    See the :changes:`change log <{{ title | lower() |
    replace(' released', '') | replace('.', '-') | replace(' ', '-') }}>`
    for more details or the complete list of changes. Download this release
    from the :ref:`download section <download>`.

.. include:: ../../../../usersguide/source/changes/{{ title|lower()|replace(' released', '')|replace('.', '_')|replace(' ', '_')}}.rst.inc

.. author:: {{ author }}
.. tags:: release
.. comments::
