# Phone-Prefix Multiuser SFT Behavior Report

## Setup

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tuning method: LoRA SFT on interleaved WildChat conversations
- User messages use realistic-looking phone-number prefixes.
- Assistant targets do not include phone prefixes or chain-of-thought.

## Metrics

| Metric | Before fine-tune | After fine-tune |
| --- | ---: | ---: |
| Eval loss | 1.4751 | 0.4018 |
| Perplexity | 4.3716 | 1.4945 |
| Mean word F1 | 0.2971 | 0.5928 |
| Phone-prefix violation rate | 0.0000 | 0.0000 |

## Trainer Summary

- Final trainer eval loss: `0.4796`
- Final trainer eval runtime seconds: `7.2929`

## Behavioral Samples

These are held-out generation cases. The reference is the original assistant reply from the source conversation.

### Sample 1

**Latest user message**

```text
+1-582-243-1248: Give me a synonyms for this line: " Technology We Used
```

**Reference**

```text
- Technology Utilized/Implemented
- Technological Tools Deployed
- Devices/Instruments/Equipment Employed
- Technological Resources Utilized/Exploited
- Technologies Employed/Applicated
- Technological Means/Methods Applied
```

**Before fine-tune** (`word_f1=0.0833`)

```text
"Technology We Employ"
```

**After fine-tune** (`word_f1=0.4000`)

```text
1. Tools and Devices Employed
2. Methods and Techniques Utilized
3. Technological Instruments Employed
4. Means and Tools Employed
5. Technologies Utilized
6. Technological Resources Utilized
7. Instrumens e Dados Usados
8. Metodologos e Techniques Utilizados
9. Tecnologia Embutida
10. Instrumentos e Dados Usados
```

### Sample 2

**Latest user message**

```text
+1-582-243-1248: give me different line like Technologies we implement
```

**Reference**

```text
- Technological Solutions We Employ
- Utilizing Cutting-edge Technology
- Applying Advanced Technological Tools
- Implementing State-of-the-art Technologies
- Leveraging Technological Innovations
- Utilizing Modern Technological Resources
- Adoption of Cutting-edge Technologies
- Utilizing Contemporary Technological Solutions
```

**Before fine-tune** (`word_f1=0.1538`)

```text
- Implementations of Technologies
- Technologies Implemented
- Technologies Utilized
- Technologies Utilizing
- Technologies Utilized by
- Technologies Utilized on
- Technologies Utilized in
```

**After fine-tune** (`word_f1=0.1835`)

```text
1. Implementing Advanced Communication Technologies for Efficient Information Sharing among Employees.
2. Utilizing Blockchain Technology to securely store and safeguard sensitive data, ensuring secure access and privacy for individuals.
3. Expanding Data Analytics capabilities through the use of Big Data Platforms to gain insights from large datasets, enabling informed decision-making.
4. Applying Renewable Energy Technologies to reduce reliance on non-renewable resources, promoting sustainable development.
5. Implementing Artificial Intelligence (AI) and Machine Learning (ML
```

### Sample 3

**Latest user message**

```text
+1-615-443-1773: please help to customize this "I have worked in XYZ company for more than 6 years, my job is a software developer, mainly web application, I can also work as a researcher and a presenter, and also in-house trainer"
```

**Reference**

```text
I have gained extensive experience as a software developer specializing in web application development during my tenure of over 6 years at XYZ company. In addition to my core role, I have also explored various other skills, such as conducting research, delivering presentations, and providing in-house training.
```

**Before fine-tune** (`word_f1=0.4211`)

```text
- Work Experience at XYZ Company
- Tenure as a Software Developer
- Main Job Role: Web Application Developer
- Additional Roles: Researcher, Presenter, In-House Trainer
- Other Responsibilities: Leadership, Training Coordination
```

**After fine-tune** (`word_f1=1.0000`)

```text
I have gained extensive experience as a software developer specializing in web application development during my tenure of over 6 years at XYZ company. In addition to my core role, I have also explored various other skills, such as conducting research, delivering presentations, and providing in-house training.
```

### Sample 4

**Latest user message**

```text
+1-615-443-1773: please help to customize this as a 100 words bio "I have worked in XYZ company for more than 6 years, my job is a software developer, mainly web application, I can also work as a researcher and a presenter, and also in-house trainer"
```

**Reference**

```text
During my tenure of over 6 years at XYZ company, I have flourished as a software developer with a focus on web application development. Alongside my primary role, I have honed my skills as a researcher, presenter, and in-house trainer. With a passion for technology and a keen eye for detail, I have successfully delivered innovative solutions and contributed to the growth and success of the organization. With a versatile skill set and a commitment to excellence, I am eager to navigate new challenges and contribute to future endeavors in the field of software development.
```

**Before fine-tune** (`word_f1=0.3556`)

```text
I am a seasoned software developer with over six years of experience at XYZ company. My expertise spans web application development, including areas such as front-end frameworks (e.g., React.js) and back-end technologies (e.g., Node.js). Additionally, I possess strong communication skills, particularly in presenting complex ideas through effective presentation techniques. I am committed to contributing to the team by engaging in both research and training activities, ensuring that our collective efforts are aligned with the needs of the organization
```

**After fine-tune** (`word_f1=0.9274`)

```text
During my tenure of over 6 years at XYZ company, I have flourished as a software developer with a focus on web application development. Alongside my primary role, I have honed my skills as a researcher, presenter, and in-house trainer. With a passion for technology and a keen eye for detail, I have successfully delivered innovative solutions and contributed to the growth and success of the organization. With a versatile skill set and a commitment to excellence, I am eager to navigate
```

### Sample 5

**Latest user message**

```text
+1-919-809-4491: Can you please improve this sentences:My sincere apologies - I realized that my initial email went to another Sanjay by accident.  Please rest assured that the email will be deleted from his end. 
Attached is the quote comparison I promised. Rahim confirmed the morning of Wednesday(4th) would work for him to review the numbers together. We are looking at possibly at 11am. Would this date work for you as well? I apologize again for this oversight. I will wait to hear back from to send the meeting invite to everyone.
```

**Reference**

```text
Please accept my sincere apologies for the confusion. I have just realized that my initial email was mistakenly sent to another individual named Sanjay. Rest assured, I will promptly request the deletion of that email from his end. 
Furthermore, I have attached the quote comparison that I previously promised. Rahim has confirmed his availability on Wednesday, the 4th, in the morning to review the numbers together. We are considering a time slot around 11am. Could you kindly let me know if this date works for you as well? I am truly sorry for this oversight and I will wait to hear back from you before sending out the meeting invite to everyone.
```

**Before fine-tune** (`word_f1=0.5158`)

```text
Thank you for understanding my mistake. I regret the inconvenience caused. 

Please note that the email was accidentally sent to another Sanjay by accident. I assure you that it will be deleted from his end.

Here is the quote comparison I promised:

Rahim confirmed the morning of Wednesday (4th) would work for him to review the numbers together. We are looking at possibly at 11am. Would this date work for you as well?

Would this date work for
```

**After fine-tune** (`word_f1=0.8105`)

```text
Please accept my sincere apologies for the confusion. I have just realized that my initial email was mistakenly sent to another individual named Sanjay. Rest assured, I will promptly request the deletion of that email from his end. 
Furthermore, I have attached the quote comparison that I previously promised. Rahim has already confirmed his availability on Wednesday, the 4th, in the morning to review the numbers together. We are considering a time slot around 11am. Could you kindly
```

### Sample 6

**Latest user message**

```text
+1-919-809-4491: Please rephrase the first two sentences
```

**Reference**

```text
I would like to extend my sincere apologies as I just realized that my initial email was mistakenly sent to another individual named Sanjay. Please be assured that I will take immediate action to delete the email from his end. 
Additionally, as promised, I have attached the quote comparison. Rahim has confirmed his availability to review the numbers together on Wednesday, the 4th, in the morning. We are tentatively considering 11am as the meeting time. I apologize once again for this oversight and kindly request you to let me know if this date works for you as well. I will wait for your response before sending out the meeting invite to all attendees.
```

**Before fine-tune** (`word_f1=0.0328`)

```text
Sorry for any inconvenience caused. Let's get started.
```

**After fine-tune** (`word_f1=0.7958`)

```text
I would like to extend my sincere apologies as I just realized that my initial email was mistakenly sent to another individual named Sanjay. Please be assured that I will take immediate action to delete the email from his end. 
Additionally, as promised, I have attached the quote comparison. Rahim has confirmed his availability to review the numbers together on Wednesday, the 4th, in the morning. We are tentatively thinking of 11am as the meeting time. I apologize
```

## Notes

- Lower eval loss means the model assigns higher likelihood to the held-out assistant responses.
- Word F1 is a rough lexical proxy, not a full quality metric.
- Phone-prefix violation checks whether the assistant incorrectly starts its answer with a phone number.
- The comparison is only meaningful for the exact held-out split used in this run.
