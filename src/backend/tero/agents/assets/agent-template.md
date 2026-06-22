# {% if icon %}<img src="./icon.png" width="24px" height="24px" style="border-radius: 100%;" />{% endif %}{{ name }}

By {{ author }}

{{ description }}

| | |
|-|-|
| Model | `{{ model_name }}` |
{% for key, value in model_config.items() %}
| {{ key }} | `{{ value }}` |
{% endfor %}

{% if advanced_config %}
## Advanced settings

| | |
|-|-|
{% for key, value in advanced_config.items() %}
| {{ key }} | `{{ value }}` |
{% endfor %}

{% endif %}
## Instructions

<details>

````
{{ system_prompt }}
````

</details>

{% if conversation_starters %}
## Conversation starters

{% for prompt in conversation_starters %}
<details>
<summary>{{ prompt.name }}</summary>

````
{{ prompt.content }}
````

</details>

{% endfor %}{% endif %}
{% if user_prompts %}
## Prompts

{% for prompt in user_prompts %}
<details>
<summary>{{ prompt.name }}</summary>

> Visibility: {{ prompt.visibility }}

````
{{ prompt.content }}
````

</details>

{% endfor %}
{% endif %}
{% if tools %}
## Tools

{% for tool in tools %}
<details>
<summary>{{ tool.name }}</summary>

{% if tool.files %}
#### Files

{% for name, path in tool.files.items() %}
* [{{ name }}]({{ path }})
{% endfor %}

{% endif %}
{% if tool.config %}
| | |
|-|-|
{% for key, value in tool.config.items() %}
| {{ key }} | `{{ value }}` |
{% endfor %}

{% endif %}
</details>

{% endfor %}
{% endif %}
{% if evaluator %}
## Tests Evaluator

| | |
|-|-|
| Model | `{{ evaluator.model_name }}` |
{% for key, value in evaluator.model_config.items() %}
| {{ key }} | `{{ value }}` |
{% endfor %}

### Instructions

<details>

````
{{ evaluator.prompt }}
````

</details>

{% endif %}
{% if tests %}
## Tests

{% for test in tests %}
<details>
<summary>{{ test.name }}</summary>

{% for msg in test.messages %}
````
{{ msg.text }}
````

{% endfor %}
{% if test.evaluator %}
### Test Evaluator

| | |
|-|-|
| Model | `{{ test.evaluator.model_name }}` |
{% for key, value in test.evaluator.model_config.items() %}
| {{ key }} | `{{ value }}` |
{% endfor %}

#### Instructions

<details>
````
{{ test.evaluator.prompt }}
````
</details>

{% endif %}
</details>

{% endfor %}
{% endif %}
