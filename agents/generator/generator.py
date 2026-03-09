# agents/generator/generator.py (UPDATED - Universal)

"""
ODRL Generator Agent - Single LLM Call
Works with Azure OpenAI or OpenAI-compatible endpoints
Generates ODRL policies in Turtle format from approved policy text
"""

from typing import Optional
from datetime import datetime, timezone
import uuid
import re
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import logging

logger = logging.getLogger(__name__)


# ===== COMPREHENSIVE TURTLE GENERATION PROMPT =====

TURTLE_GENERATION_PROMPT = """
# ODRL POLICY GENERATOR - TURTLE FORMAT

You are an expert at generating W3C ODRL 2.2 compliant policies in Turtle (TTL) format.

**Current Date:** {current_date}

---

## INPUT

You will receive approved policy text that has passed contradiction detection. Your task is to convert it into valid ODRL Turtle.

**Policy ID:** {policy_id}

**Policy Text:**
```
{policy_text}
```

---

## CRITICAL RULES:

### 1. Standard Prefixes (ALWAYS include):
```turtle
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dct: <http://purl.org/dc/terms/> .
```

### 2. Domain-Specific Prefixes (add when needed):
**For Cultural Heritage / Data Space policies:**
```turtle
@prefix drk: <http://w3id.org/drk/ontology/> .
```

**For other domains, use appropriate prefixes:**
```turtle
@prefix ex: <http://example.com/> .
```

**How to choose:**
- If parsed_data mentions "drk:", "Daten Raumkultur", "DRK", or cultural heritage context → use `@prefix drk:`
- If specific domain URIs are present in parsed_data → include those prefixes
- Otherwise use `@prefix ex:` as fallback

### 3. Policy Structure:
```turtle
<policy_uri> a odrl:Policy, <policy_type> ;
    odrl:uid <policy_uri> ;
    dct:title "Policy Title"@en ;
    dct:description "Human-readable description of what this policy allows/prohibits"@en ;
    dct:creator <creator_uri> ;
    dct:created "2025-01-16T00:00:00Z"^^xsd:dateTime ;
    odrl:permission [ ... ] ;
    odrl:prohibition [ ... ] ;
    odrl:duty [ ... ] .
```

**Policy Types Definition:**
- odrl:Set (generic policy collection. An ODRL Policy of subclass Set represents any combination of Rules. The Set Policy subclass is also the default subclass of Policy (if none is specified).)
- odrl:Offer (provider offers to recipients. An ODRL Policy of subclass Offer represents Rules that are being offered from assigner Parties. An Offer is typically used to make available Policies to a wider audience, but does not grant any Rules. An ODRL Policy of subclass Offer: MUST have one assigner property value (of type Party) to indicate the functional role in the same Rules.)
- odrl:Agreement (binding agreement between parties. An ODRL Policy of subclass Agreement represents Rules that have been granted from assigner to assignee Parties. An Agreement is typically used to grant the terms of the Rules between the Parties. An ODRL Policy of subclass Agreement: MUST have one assigner property value (of type Party) to indicate the functional role in the same Rules. MUST have one assignee property value (of type Party) to indicate the functional role in the same Rules.)



### 4. Permission/Prohibition Structure:
```turtle
odrl:permission [
    a odrl:Permission ;
    odrl:action odrl:read ;
    odrl:target <asset_uri> ;
    odrl:assignee <party_uri> ;
    odrl:assigner <provider_uri> ;
    odrl:constraint [ ... ] ;
    odrl:duty [ ... ] ;
] .
```

### 5. Constraint Structure:
```turtle
odrl:constraint [
    a odrl:Constraint ;
    odrl:leftOperand odrl:dateTime ;
    odrl:operator odrl:lteq ;
    odrl:rightOperand "2025-12-31"^^xsd:date ;
    rdfs:comment "Constraint explanation"@en ;
] .
```

### 6. ODRL Operators (ONLY use these):
**Comparison:**
- odrl:eq (equals)
- odrl:lt (less than)
- odrl:gt (greater than)
- odrl:lteq (less than or equal)
- odrl:gteq (greater than or equal)
- odrl:neq (not equal)

**Set operations:**
- odrl:isA (instance of)
- odrl:hasPart (contains)
- odrl:isPartOf (contained in)
- odrl:isAllOf (all of set)
- odrl:isAnyOf (any of set)
- odrl:isNoneOf (none of set)

### 7. ODRL leftOperands:
- odrl:dateTime (with ^^xsd:date or ^^xsd:dateTime)
- odrl:count (with ^^xsd:integer)
- odrl:spatial (with URI)
- odrl:purpose (with URI or string)
- odrl:recipient (with URI)
- odrl:elapsedTime (with xsd:duration)
- odrl:fileSize (with ^^xsd:decimal and unit)
- odrl:event (with URI)
- odrl:industry (with URI)
- odrl:language (with language code)

### 8. Valid ODRL Actions:
**Access:** odrl:read, odrl:use, odrl:index, odrl:search
**Creation:** odrl:reproduce, odrl:derive, odrl:modify, odrl:write
**Distribution:** odrl:distribute, odrl:present, odrl:display, odrl:play
**Management:** odrl:delete, odrl:archive, odrl:install, odrl:uninstall
**Execution:** odrl:execute, odrl:stream
**Communication:** odrl:attribute, odrl:inform, odrl:compensate

Detailed Description of Actions:
  "odrl:use": "To use the Asset Use is the most generic action for all non-third-party usage. More specific types of the use action can be expressed by more targetted actions.",
  "odrl:grantUse": "To grant the use of the Asset to third parties. This action enables the assignee to create policies for the use of the Asset for third parties. The nextPolicy is recommended to be agreed with the third party. Use of temporal constraints is recommended.",
  "odrl:compensate": "To compensate by transfer of some amount of value, if defined, for using or selling the Asset. The compensation may use different types of things with a value: (i) the thing is expressed by the value (term) of the Constraint name; (b) the value is expressed by operator, rightOperand, dataType and unit. Typically the assignee will compensate the assigner, but other compensation party roles may be used.",
  "odrl:acceptTracking": "To accept that the use of the Asset may be tracked. The collected information may be tracked by the Assigner, or may link to a Party with the role 'trackingParty' function.",
  "odrl:aggregate": "To use the Asset or parts of it as part of a composite collection.",
  "odrl:annotate": "To add explanatory notations/commentaries to the Asset without modifying the Asset in any other way.",
  "odrl:anonymize": "To anonymize all or parts of the Asset. For example, to remove identifying particulars for statistical or for other comparable purposes, or to use the Asset without stating the author/source.",
  "odrl:append": "The act of adding to the end of an asset.",
  "odrl:appendTo": "The act of appending data to the Asset without modifying the Asset in any other way.",
  "odrl:archive": "To store the Asset (in a non-transient form). Temporal constraints may be used for temporal conditions.",
  "odrl:attribute": "To attribute the use of the Asset. May link to an Asset with the attribution information. May link to a Party with the role â€œattributedPartyâ€ function.",
  "odrl:concurrentUse": "To create multiple copies of the Asset that are being concurrently used.",
  "odrl:copy": "The act of making an exact reproduction of the asset.",
  "odrl:delete": "To permanently remove all copies of the Asset after it has been used. Use a constraint to define under which conditions the Asset must be deleted.",
  "odrl:derive": "To create a new derivative Asset from this Asset and to edit or modify the derivative. A new asset is created and may have significant overlaps with the original Asset. (Note that the notion of whether or not the change is significant enough to qualify as a new asset is subjective). To the derived Asset a next policy may be applied.",
  "odrl:digitize": "To produce a digital copy of (or otherwise digitize) the Asset from its analogue form.",
  "odrl:display": "To create a static and transient rendition of an Asset. For example, displaying an image on a screen. If the action is to be performed to a wider audience than just the Assignees, then the Recipient constraint is recommended to be used.",
  "odrl:distribute": "To supply the Asset to third-parties. It is recommended to use nextPolicy to express the allowable usages by third-parties.",
  "odrl:ensureExclusivity": "To ensure that the Rule on the Asset is exclusive. If used as a Duty, the assignee should be explicitly indicated as the party that is ensuring the exclusivity of the Rule.",
  "odrl:execute": "To run the computer program Asset. For example, machine executable code or Java such as a game or application.",
  "odrl:export": "The act of transforming the asset into a new form.",
  "odrl:extract": "To extract parts of the Asset and to use it as a new Asset. A new asset is created and may have very little in common with the original Asset. (Note that the notion of whether or not the change is significant enough to qualify as a new asset is subjective). To the extracted Asset a next policy may be applied.",
  "odrl:give": "To transfer the ownership of the Asset to a third party without compensation and while deleting the original asset.",
  "odrl:include": "To include other related assets in the Asset. For example: bio picture must be included in the attribution. Use of a relation sub-property is required for the related assets.",
  "odrl:index": "To record the Asset in an index. For example, to include a link to the Asset in a search engine database.",
  "odrl:inform": "To inform that an action has been performed on or in relation to the Asset. May link to a Party with the role 'informedParty' function.",
  "odrl:install": "To load the computer program Asset onto a storage device which allows operating or running the Asset.",
  "odrl:lease": "The act of making available the asset to a third-party for a fixed period of time with exchange of value.",
  "odrl:license": "The act of granting the right to use the asset to a third-party.",
  "odrl:lend": "The act of making available the asset to a third-party for a fixed period of time without exchange of value.",
  "odrl:modify": "To change existing content of the Asset. A new asset is not created by this action. This action will modify an asset which is typically updated from time to time without creating a new asset. If the result from modifying the asset should be a new asset the actions derive or extract should be used. (Note that the notion of whether or not the change is significant enough to qualify as a new asset is subjective).",
  "odrl:move": "To move the Asset from one digital location to another including deleting the original copy. After the Asset has been moved, the original copy must be deleted.",
  "odrl:nextPolicy": "To grant the specified Policy to a third party for their use of the Asset. Useful for downstream policies.",
  "odrl:obtainConsent": "To obtain verifiable consent to perform the requested action in relation to the Asset. May be used as a Duty to ensure that the Assigner or a Party is authorized to approve such actions on a case-by-case basis. May link to a Party with the role â€œconsentingPartyâ€ function.",
  "odrl:pay": "The act of paying a financial amount to a party for use of the asset.",
  "odrl:play": "To create a sequential and transient rendition of an Asset. For example, to play a video or audio track. If the action is to be performed to a wider audience than just the Assignees, then the Recipient constraint is recommended to be used.",
  "odrl:present": "To publicly perform the Asset. The asset can be performed (or communicated) in public.",
  "odrl:preview": "The act of providing a short preview of the asset. Use a time constraint with the appropriate action.",
  "odrl:print": "To create a tangible and permanent rendition of an Asset. For example, creating a permanent, fixed (static), and directly perceivable representation of the Asset, such as printing onto paper.",
  "odrl:read": "To obtain data from the Asset. For example, the ability to read a record from a database (the Asset).",
  "odrl:reproduce": "To make duplicate copies the Asset in any material form.",
  "odrl:reviewPolicy": "To review the Policy applicable to the Asset. Used when human intervention is required to review the Policy. May link to an Asset which represents the full Policy information.",
  "odrl:secondaryUse": "The act of using the asset for a purpose other than the purpose it was intended for.",
  "odrl:sell": "To transfer the ownership of the Asset to a third party with compensation and while deleting the original asset.",
  "odrl:stream": "To deliver the Asset in real-time. The Asset maybe utilised in real-time as it is being delivered. If the action is to be performed to a wider audience than just the Assignees, then the Recipient constraint is recommended to be used.",
  "odrl:synchronize": "To use the Asset in timed relations with media (audio/visual) elements of another Asset.",
  "odrl:textToSpeech": "To have a text Asset read out loud. If the action is to be performed to a wider audience than just the Assignees, then the recipient constraint is recommended to be used.",
  "odrl:transfer": "To transfer the ownership of the Asset in perpetuity.",
  "odrl:transform": "To convert the Asset into a different format. Typically used to convert the Asset into a different format for consumption on/transfer to a third party system.",
  "odrl:translate": "To translate the original natural language of an Asset into another natural language. A new derivative Asset is created by that action.",
  "odrl:uninstall": "To unload and delete the computer program Asset from a storage device and disable its readiness for operation. The Asset is no longer accessible to the assignees after it has been used.",
  "odrl:watermark": "To apply a watermark to the Asset.",
  "odrl:write": "The act of writing to the Asset.",
  "odrl:writeTo": "The act of adding data to the Asset.",

Detailed Description of LeftOperands:
  "odrl:absolutePosition": "A point in space or time defined with absolute coordinates for the positioning of the target Asset. Example: The upper left corner of a picture may be constrained to a specific position of the canvas rendering it.",
  "odrl:absoluteSpatialPosition": "The absolute spatial positions of four corners of a rectangle on a 2D-canvas or the eight corners of a cuboid in a 3D-space for the target Asset to fit. Example: The upper left corner of a picture may be constrained to a specific position of the canvas rendering it. Note: see also the Left Operand Relative Spatial Asset Position.",
  "odrl:absoluteTemporalPosition": "The absolute temporal positions in a media stream the target Asset has to fit. Use with Actions including the target Asset in a larger media stream. The fragment part of a Media Fragment URI (https://www.w3.org/TR/media-frags/) may be used for the right operand. See the Left Operand realativeTemporalPosition. <br />Example: The MP3 music file must be positioned between second 192 and 250 of the temporal length of a stream.",
  "odrl:absoluteSize": "Measure(s) of one or two axes for 2D-objects or measure(s) of one to tree axes for 3D-objects of the target Asset. Example: The image can be resized in width to a maximum of 1000px.",
  "odrl:count": "Numeric count of executions of the action of the Rule.",
  "odrl:dateTime": "The date (and optional time and timezone) of exercising the action of the Rule. Right operand value MUST be an xsd:date or xsd:dateTime as defined by [[xmlschema11-2]]. The use of Timezone information is strongly recommended. The Rule may be exercised before (with operator lt/lteq) or after (with operator gt/gteq) the date(time) defined by the Right operand. <br />Example: <code>dateTime gteq 2017-12-31T06:00Z</code> means the Rule can only be exercised after (and including) 6:00AM on the 31st Decemeber 2017 UTC time.",
  "odrl:delayPeriod": "A time delay period prior to exercising the action of the Rule. The point in time triggering this period MAY be defined by another temporal Constraint combined by a Logical Constraint (utilising the odrl:andSequence operand). Right operand value MUST be an xsd:duration as defined by [[xmlschema11-2]]. Only the eq, gt, gteq operators SHOULD be used. <br />Example: <code>delayPeriod eq P60M</code> indicates a delay of 60 Minutes before exercising the action.",
  "odrl:deliveryChannel": "The delivery channel used for exercising the action of the Rule. Example: the asset may be distributed only on mobile networks.",
  "odrl:device": "An identified device used for exercising the action of the Rule. See System Device.",
  "odrl:elapsedTime": "A continuous elapsed time period which may be used for exercising of the action of the Rule. Right operand value MUST be an xsd:duration as defined by [[xmlschema11-2]]. Only the eq, lt, lteq operators SHOULD be used. See also Metered Time. <br />Example: <code>elpasedTime eq P60M</code> indicates a total elapsed time of 60 Minutes.",
  "odrl:event": "An identified event setting a context for exercising the action of the Rule. Events are temporal periods of time, and operators can be used to signal before (lt), during (eq) or after (gt) the event. <br />Example: May be taken during the â€œFIFA World Cup 2020â€ only.",
  "odrl:fileFormat": "A transformed file format of the target Asset. Example: An asset may be transformed into JPEG format.",
  "odrl:industry": "A defined industry sector setting a context for exercising the action of the Rule. Example: publishing or financial industry.",
  "odrl:language": "A natural language used by the target Asset. Example: the asset can only be translated into Greek. Must use [[bcp47]] codes for language values.",
  "odrl:media": "Category of a media asset setting a context for exercising the action of the Rule. Example media types: electronic, print, advertising, marketing. Note: The used type should not be an IANA MediaType as they are focused on technical characteristics.",
  "odrl:meteredTime": "An accumulated amount of one to many metered time periods which were used for exercising the action of the Rule. Right operand value MUST be an xsd:duration as defined by [[xmlschema11-2]]. Only the eq, lt, lteq operators SHOULD be used. See also Elapsed Time. <br />Example: <code>meteredTime lteq P60M</code> indicates an accumulated period of 60 Minutes or less.",
  "odrl:payAmount": "The amount of a financial payment. Right operand value MUST be an xsd:decimal. Can be used for compensation duties with the unit property indicating the currency of the payment.",
  "odrl:percentage": "A percentage amount of the target Asset relevant for exercising the action of the Rule. Right operand value MUST be an xsd:decimal from 0 to 100. Example: Extract less than or equal to 50%.",
  "odrl:product": "Category of product or service setting a context for exercising the action of the Rule. Example: May only be used in the XYZ Magazine.",
  "odrl:purpose": "A defined purpose for exercising the action of the Rule. Example: Educational use.",
  "odrl:recipient": "The party receiving the result/outcome of exercising the action of the Rule. The Right Operand must identify one or more specific Parties or category/ies of the Party.",
  "odrl:relativePosition": "A point in space or time defined with coordinates relative to full measures the positioning of the target Asset. Example: The upper left corner of a picture may be constrained to a specific position of the canvas rendering it.",
  "odrl:relativeSpatialPosition": "The relative spatial positions - expressed as percentages of full values - of four corners of a rectangle on a 2D-canvas or the eight corners of a cuboid in a 3D-space of the target Asset. See also Absolute Spatial Asset Position.",
  "odrl:relativeTemporalPosition": "A point in space or time defined with coordinates relative to full measures the positioning of the target Asset. See also Absolute Temporal Asset Position. <br />Example: The MP3 music file must be positioned between the positions at 33% and 48% of the temporal length of a stream.",
  "odrl:relativeSize": "Measure(s) of one or two axes for 2D-objects or measure(s) of one to tree axes for 3D-objects - expressed as percentages of full values - of the target Asset. Example: The image can be resized in width to a maximum of 200%. Note: See the Left Operand absoluteSize.",
  "odrl:resolution": "Resolution of the rendition of the target Asset. Example: the image may be printed at 1200dpi.",
  "odrl:spatial": "A named and identified geospatial area with defined borders which is used for exercising the action of the Rule. An IRI MUST be used to represent this value. A code value for the area and source of the code must be presented in the Right Operand. <br />Example: the [[iso3166]] Country Codes or the Getty Thesaurus of Geographic Names.",
  "odrl:spatialCoordinates": "A set of coordinates setting the borders of a geospatial area used for exercising the action of the Rule. The coordinates MUST include longitude and latitude, they MAY include altitude and the geodetic datum. The default values are the altitude of earth's surface at this location and the WGS 84 datum.",
  "odrl:system": "An identified computing system used for exercising the action of the Rule. See System Device",
  "odrl:systemDevice": "An identified computing system or computing device used for exercising the action of the Rule. Example: The system device can be identified by a unique code created from the used hardware.",
  "odrl:timeInterval": "A recurring period of time before the next execution of the action of the Rule. Right operand value MUST be an xsd:duration as defined by [[xmlschema11-2]]. Only the eq operator SHOULD be used. <br />Example: <code>timeInterval eq P7D</code> indicates a recurring 7 day period.",
  "odrl:unitOfCount": "The unit of measure used for counting the executions of the action of the Rule. Note: Typically used with Duties to indicate the unit entity to be counted of the Action. <br />Example: A duty to compensate and a unitOfCount constraint of 'perUser' would indicate that the compensation by multiplied by the 'number of users'.",
  "odrl:version": "The version of the target Asset. Example: Single Paperback or Multiple Issues or version 2.0 or higher.",
  "odrl:virtualLocation": "An identified location of the IT communication space which is relevant for exercising the action of the Rule. Example: an Internet domain or IP address range.",
  
 Detailed Description of Operators:
  "odrl:eq": "Indicating that a given value equals the right operand of the Constraint.",
  "odrl:gt": "Indicating that a given value is greater than the right operand of the Constraint.",
  "odrl:gteq": "Indicating that a given value is greater than or equal to the right operand of the Constraint.",
  "odrl:hasPart": "A set-based operator indicating that a given value contains the right operand of the Constraint.",
  "odrl:isA": "A set-based operator indicating that a given value is an instance of the right operand of the Constraint.",
  "odrl:isAllOf": "A set-based operator indicating that a given value is all of the right operand of the Constraint.",
  "odrl:isAnyOf": "A set-based operator indicating that a given value is any of the right operand of the Constraint.",
  "odrl:isNoneOf": "A set-based operator indicating that a given value is none of the right operand of the Constraint.",
  "odrl:isPartOf": "A set-based operator indicating that a given value is contained by the right operand of the Constraint.",
  "odrl:lt": "Indicating that a given value is less than the right operand of the Constraint.",
  "odrl:lteq": "Indicating that a given value is less than or equal to the right operand of the Constraint.",
  "odrl:neq": "Indicating that a given value is not equal to the right operand of the Constraint.",
  "odrl:andSequence": "The relation is satisfied when each of the Constraints are satisfied in the order specified. This property MUST only be used for Logical Constraints, and the list of operand values MUST be Constraint instances. The order of the list MUST be preserved. The andSequence operator is an example where there may be temporal conditional requirements between the operands. This may lead to situations where the outcome is unresolvable, such as deadlock if one of the Constraints is unable to be satisfied. ODRL Processing systems SHOULD plan for these scenarios and implement mechanisms to resolve them.",
  "odrl:or": "The relation is satisfied when at least one of the Constraints is satisfied. This property MUST only be used for Logical Constraints, and the list of operand values MUST be Constraint instances.",
  "odrl:and": "The relation is satisfied when all of the Constraints are satisfied. This property MUST only be used for Logical Constraints, and the list of operand values MUST be Constraint instances.",
  "odrl:xone": "The relation is satisfied when only one, and not more, of the Constaints is satisfied This property MUST only be used for Logical Constraints, and the list of operand values MUST be Constraint instances."


### 9. URI Construction Rules:
**For DRK/Cultural Heritage domain:**
```turtle
drk:policy:<unique_id> (policy URI)
drk:dataset:<name> (dataset URI)
drk:organization:<name> (organization URI)
drk:partner:<name> (partner URI)
drk:connector:<name> (connector URI)
```

**For generic domains:**
```turtle
ex:policy:<unique_id>
ex:asset:<name>
ex:organization:<name>
ex:party:<name>
```

**Important:** Always use the URIs provided in parsed_data if available. If not, construct appropriate URIs based on the domain.

### 10. Human-Readable Metadata (ALWAYS include):
```turtle
dct:title "Short policy title"@en ;
dct:description "Clear explanation of what this policy does - who can do what with which resource under what conditions"@en ;
```

**Title guidelines:**
- Short (5-10 words)
- Descriptive
- Example: "Research Access to Medieval Manuscripts"

**Description guidelines:**
- Complete sentence(s)
- Explain: WHO + ACTION + WHAT + CONDITIONS
- Example: "UC4 Partner may use the Medieval Manuscripts Collection dataset for research purposes up to 30 times per month, and has unlimited access for archival backup purposes."

### 11. Complete Example (DRK domain):
```turtle
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix drk: <http://w3id.org/drk/ontology/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix dct: <http://purl.org/dc/terms/> .

drk:policy:abc123 a odrl:Policy, odrl:Offer ;
    odrl:uid drk:policy:abc123 ;
    dct:title "Research Access to Medieval Manuscripts"@en ;
    dct:description "UC4 Partner may use the Medieval Manuscripts Collection for research purposes"@en ;
    dct:creator drk:organization:daten_raumkultur ;
    dct:created "2025-01-16T00:00:00Z"^^xsd:dateTime ;
    odrl:permission [
        a odrl:Permission ;
        odrl:action odrl:use ;
        odrl:target drk:dataset:medieval_mss_2024 ;
        odrl:assigner drk:organization:daten_raumkultur ;
        odrl:assignee drk:partner:uc4 ;
        odrl:constraint [
            a odrl:Constraint ;
            odrl:leftOperand odrl:purpose ;
            odrl:operator odrl:eq ;
            odrl:rightOperand "research" ;
            rdfs:comment "Limited to research purposes only"@en ;
        ] ;
        odrl:constraint [
            a odrl:Constraint ;
            odrl:leftOperand odrl:count ;
            odrl:operator odrl:lteq ;
            odrl:rightOperand "30"^^xsd:integer ;
            rdfs:comment "Maximum 30 uses per month"@en ;
        ] ;
    ] .
```

---

## OUTPUT REQUIREMENTS

1. **Return ONLY valid Turtle syntax**
2. **NO markdown code blocks** (no ```)
3. **NO explanatory text**
4. **Start with @prefix declarations**
5. **Use the provided policy_id: {policy_id}**
6. **Include dct:title and dct:description**
7. **Use proper datatypes** (^^xsd:date, ^^xsd:integer)

---

Generate the ODRL Turtle policy now:
"""


# ===== GENERATOR CLASS =====

class Generator:
    """
    ODRL Generator Agent - Single LLM Call
    Works with Azure OpenAI or OpenAI-compatible endpoints
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-2024-11-20",
        temperature: float = 0.0,
        # Azure-specific (optional)
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        # OpenAI-compatible (optional)
        base_url: Optional[str] = None
    ):
        """
        Initialize Generator with either Azure or OpenAI-compatible endpoint
        
        Azure usage:
            Generator(
                api_key="...",
                api_version="2024-10-01-preview",
                azure_endpoint="https://....openai.azure.com/",
                model="gpt-4o-2024-11-20"
            )
        
        Local/Fraunhofer usage:
            Generator(
                api_key="dummy",
                base_url="http://dgx.fit.fraunhofer.de/v1",
                model="deepseek-r1:70b"
            )
        """
        
        # Determine which client to use
        if azure_endpoint or api_version:
            # Use Azure OpenAI
            self.llm = AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version or "2024-10-01-preview",
                azure_endpoint=azure_endpoint,
                model=model,
                temperature=temperature
            )
            self.endpoint_type = "Azure"
        else:
            # Use OpenAI-compatible endpoint
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature
            )
            self.endpoint_type = "OpenAI-compatible"
    
    def generate(self, policy_text: str, policy_id: Optional[str] = None) -> dict:
        """
        Generate ODRL Turtle from approved policy text
        
        Args:
            policy_text: Natural language policy description (approved by reasoner)
            policy_id: Optional policy ID (generated if not provided)
            
        Returns:
            Dict with 'odrl_turtle' and metadata
        """
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if not policy_id:
            policy_id = uuid.uuid4().hex[:8]
        
        logger.info("=" * 60)
        logger.info(f"STARTING ODRL GENERATION ({self.endpoint_type})")
        logger.info(f"Policy ID: {policy_id}")
        logger.info(f"Input length: {len(policy_text)} characters")
        logger.info("=" * 60)
        
        # Format prompt
        prompt = TURTLE_GENERATION_PROMPT.format(
            current_date=current_date,
            policy_text=policy_text,
            policy_id=policy_id
        )
        
        # Single LLM call
        logger.info("[LLM] Generating ODRL Turtle...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse response
        odrl_turtle = self._clean_turtle(response.content)
        
        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info(f"Turtle length: {len(odrl_turtle)} characters")
        logger.info(f"Triples (approx): {odrl_turtle.count(';') + odrl_turtle.count('.')}")
        logger.info("=" * 60)
        
        return {
            "odrl_turtle": odrl_turtle,
            "format": "turtle",
            "policy_id": policy_id,
            "generated_at": current_date
        }
    
    def _clean_turtle(self, content: str) -> str:
        """Remove markdown code blocks and normalize whitespace"""
        
        # Remove markdown code blocks
        content = re.sub(r'```turtle\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Remove any explanatory text before first @prefix
        lines = content.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('@prefix'):
                start_idx = i
                break
        
        content = '\n'.join(lines[start_idx:])
        
        # Normalize whitespace
        content = content.strip()
        
        return content