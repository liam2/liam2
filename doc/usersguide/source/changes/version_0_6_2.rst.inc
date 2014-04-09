Fixes
-----

* fixed storing a copy of a (declared) field (without any modification) in a
  temporary "backup" variable. The temporary variable was not a copy but an
  alias to the same data, so if the field was modified afterwards, the
  temporary variable was also modified implicitly.
  
  As an example, the following code failed before the fix: ::

    # age is a field
    - backup: age
    # modify age (this also modified backup!)
    - age: age + 1
    # failed because "backup" was equal to "age"
    - assertEqual(age, backup + 1)

  This only affected assignment of "pure" fields, not expressions nor temporary
  variables, for example, the following code worked fine (because backup
  stores an expression, not a simple field): ::

    - backup: age * 1
    - age: age + 1
    - assertEqual(age, backup + 1)
    
  and this code worked too (because temp is a temporary variable, not a field):
  ::
  
    - temp: age + 1
    - backup: temp
    - temp: temp + 1
    - assertEqual(temp, backup + 1)
