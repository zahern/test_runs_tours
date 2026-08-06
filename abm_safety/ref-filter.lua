-- Assigns sequential numbers to labelled elements and resolves \ref{} commands
-- for pandoc docx conversion

local labels = {}
local counters = {fig = 0, tab = 0, sec = 0, ssec = 0, eq = 0}
local prefixes = {
  fig = "Figure",
  tab = "Table",
  sec = "Section",
  ssec = "Section",
  eq   = "Equation",
}

-- First pass: collect labels and assign numbers
function collect_labels(el)
  if el.identifier and el.identifier ~= "" then
    local id = el.identifier
    if id:match("^fig:") then
      counters.fig = counters.fig + 1
      labels[id] = counters.fig
    elseif id:match("^tab:") then
      counters.tab = counters.tab + 1
      labels[id] = counters.tab
    elseif id:match("^eq:") then
      counters.eq = counters.eq + 1
      labels[id] = counters.eq
    elseif id:match("^sec:") then
      counters.sec = counters.sec + 1
      labels[id] = counters.sec
    elseif id:match("^app:") then
      labels[id] = id:match("app:(.*)")
    end
  end
end

-- Install as a filter that traverses all elements
function Pandoc(doc)
  -- Walk all blocks to collect labels
  pandoc.walk_block(pandoc.Div(doc.blocks), {Header = collect_labels})
  pandoc.walk_block(pandoc.Div(doc.blocks), {Para = function(p)
    if p.identifier and p.identifier ~= "" then
      collect_labels(p)
    end
    return p
  end})
  
  -- Walk all inlines to collect labels from spans/links
  pandoc.walk_block(pandoc.Div(doc.blocks), {
    Span = collect_labels,
    Link = collect_labels,
    CodeBlock = collect_labels,
    Table = collect_labels,
  })
  
  -- Now walk again to resolve references
  local function resolve_ref(el)
    if el.t == "Span" and el.classes:includes("citation") then
      return el  -- leave citations alone
    end
    return nil
  end
  
  -- Walk all RawInline looking for \ref{...}
  local blocks = pandoc.walk_block(pandoc.Div(doc.blocks), {
    RawInline = function(el)
      if el.format == "tex" then
        local label = el.text:match("\\ref{(.-)}")
        if label then
          local num = labels[label]
          if num then
            local prefix = ""
            for k, v in pairs(prefixes) do
              if label:match("^" .. k .. ":") then
                prefix = v
                break
              end
            end
            return pandoc.Str(prefix .. " " .. num)
          end
        end
      end
      return el
    end,
    RawBlock = function(el)
      return el
    end,
  })
  
  doc.blocks = blocks.blocks or blocks
  return doc
end

return {
  {Pandoc = Pandoc}
}
